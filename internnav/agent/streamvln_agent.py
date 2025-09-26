import random
import time
import copy
import numpy as np
import torch
from gym import spaces
import transformers
from collections import OrderedDict
from PIL import Image, ImageFile
import re
import itertools

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg
from internnav.configs.model.base_encoders import ModelCfg
from internnav.evaluator.utils.models import batch_obs
from internnav.model import get_config, get_policy
from internnav.model.basemodel.LongCLIP.model import longclip
from internnav.model.basemodel.rdp.utils import (
    FixedLengthStack,
    compute_actions,
    get_delta,
    map_action_to_2d,
    normalize_data,
    quat_to_euler_angles,
    to_local_coords_batch,
)
from internnav.model.utils.bert_token import BertTokenizer
from internnav.model.utils.feature_extract import (
    extract_image_features,
    extract_instruction_tokens,
)
from internnav.utils import common_log_util
from internnav.utils.common_log_util import common_logger as log
from internnav.model.utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN


def dict_to_cuda(input_dict, device):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.to(device)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.to(device) for ele in v]
        # elif (
        #     isinstance(input_dict[k], list)
        #     and len(input_dict[k]) > 0
        #     and isinstance(input_dict[k][0], Det3DDataElement)
        # ):
        #     input_dict[k] = [ele.to(device) for ele in v]

    return input_dict

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@Agent.register('streamvln')
class StreamvlnAgent(Agent):
    observation_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(256, 256, 1),
        dtype=np.float32,
    )

    def __init__(self, config: AgentCfg):
        super().__init__(config)
        set_random_seed(0)
        self._model_settings = self.config.model_settings
        self._model_settings = ModelCfg(**self._model_settings)
        env_num = getattr(self._model_settings, 'env_num', 1)
        proc_num = getattr(self._model_settings, 'proc_num', 1)
        self.device = torch.device('cuda', 0)

        local_rank = 0
        model_max_length = 4096
        num_history = 8

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.ckpt_path,
                                                            model_max_length=model_max_length,
                                                            padding_side="right")

        # from transformers import LlavaConfig
        # config = LlavaConfig.from_pretrained(config.ckpt_path)
        llava_config = transformers.AutoConfig.from_pretrained(config.ckpt_path)
        policy = get_policy(self._model_settings.policy_name)
        self.policy = policy.from_pretrained(
                    config.ckpt_path,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    config=llava_config,
                    load_in_4bit=True, 
                    device_map={"": local_rank},
                    low_cpu_mem_usage=True,
                    )
        self.policy.model.num_history = num_history
        self.policy.requires_grad_(False)
        self.policy.eval()
        self.policy.reset(1)

        self.image_processor = self.policy.get_vision_tower().image_processor
        prompt = f"<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3]
        })
        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]
        self.num_frames = 32
        self.num_future_steps = 4
        self.num_history = num_history
 
        # step required
        self._env_nums = env_num * proc_num
        self._reset_ls = set()

        self._reset()

    def reset(self, reset_ls=None):
        if reset_ls is None:
            reset_ls = [i for i in range(self._env_nums)]
        self._reset_ls.update(reset_ls)
        log.debug(f'new reset_ls: {self._reset_ls}')

    def _reset(self):
        self.rgb_list = []
        self.depth_list = []
        self.depth_images_list = []
        self.pose_list = []
        self.intrinsic_list = []
        self.time_ids = []
        self.action_seq = []
        self.step_id = 0
        self.output_ids = None
        self.past_key_values = None
        self._reset_ls = set()

        for i in self._reset_ls:
            self.policy.reset_for_env(i)

    @property
    def _need_reset(self):
        return (len(self.action_seq) == 0 and len(self._reset_ls) > 0) or (len(self._reset_ls) >= self._env_nums)

    def preprocess_qwen(self, sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.",add_system: bool = False):
        # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
        roles = {"human": "user", "gpt": "assistant"}
        # import ipdb; ipdb.set_trace()
        # Add image tokens to tokenizer as a special tokens
        # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
        tokenizer = copy.deepcopy(tokenizer)
        # When there is actually an image, we add the image tokens as a special token
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = tokenizer.additional_special_tokens_ids
        # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
        unmask_tokens_idx =  [198, im_start, im_end]
        nl_tokens = tokenizer("\n").input_ids

        # Reset Qwen chat templates so that it won't include system message every time we apply
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        # _system = tokenizer("system").input_ids + nl_tokens
        # _user = tokenizer("user").input_ids + nl_tokens
        # _assistant = tokenizer("assistant").input_ids + nl_tokens

        # Apply prompt templates
        conversations = []
        input_ids = []
        for i, source in enumerate(sources):
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else: 
                source[0]["value"] = f"{prompt}."
            if roles[source[0]["from"]] != roles["human"]:
                # Skip the first one if it is not from human
                source = source[1:]

            input_id, target = [], []

            # import ipdb; ipdb.set_trace()
            # New version, use apply chat template
            # Build system message for each sentence
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])

            for conv in source:
                # Make sure llava data can load
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role =  roles.get(role, role)
                
                conv = [{"role" : role, "content" : content}]
                # import ipdb; ipdb.set_trace()
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id
            

            # assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
                    
            input_ids.append(input_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids,  conversations # tensor(bs x seq_len)

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def inference(self, obs):
        if self._need_reset:
            # import ipdb; ipdb.set_trace()
            self._reset()
            log.debug(f'model reset_ls: {self._reset_ls}')
        
        self.time_ids.append(self.step_id)

        observations = obs[0]
        rgb = observations["rgb"]
        image = Image.fromarray(rgb).convert('RGB')
        image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
        self.rgb_list.append(image)

        if len(self.action_seq) == 0:
            if self.output_ids is None:
                sources = copy.deepcopy(self.conversation)
                sources[0]["value"] = sources[0]["value"].replace(' Where should you go next to stay on track?', f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
                if self.step_id != 0 :
                    sources[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
                sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN+'\n', '')
                sources[0]["value"] = sources[0]["value"].replace('<instruction>.', observations['instruction'])
                add_system = True
                print(self.step_id, sources[0]["value"])
            else:
                sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                add_system = False

            input_ids, conversations = self.preprocess_qwen([sources], self.tokenizer, True, add_system=add_system)
            if self.output_ids is not None:
                input_ids = torch.cat([self.output_ids,input_ids.to(self.output_ids.device)], dim=1)

            images = self.rgb_list[-1:]
            # import ipdb; ipdb.set_trace()
            if self.step_id != 0 and self.step_id % self.num_frames == 0:
                # import ipdb; ipdb.set_trace()
                if self.num_history is None:
                    history_ids = slice(0, self.time_ids[-1], self.num_future_steps)
                else:
                    history_ids = slice(0, self.time_ids[-1], (self.time_ids[-1] // self.num_history))
                images = self.rgb_list[history_ids] + images
                    
            input_dict = {'images':torch.stack(images).unsqueeze(0), 'depths':None, \
                            'poses':None, 'intrinsics':None, \
                            'inputs':input_ids, 'env_id':0, 'time_ids':[self.time_ids],'task_type':[0]}
                
            input_dict = dict_to_cuda(input_dict, self.device)
            
            for key, value in input_dict.items():
                if key in ['images']:
                    input_dict[key] = input_dict[key].to(torch.bfloat16)
            
            outputs = self.policy.generate(**input_dict, do_sample=False, num_beams=1, max_new_tokens=10000, use_cache=True, return_dict_in_generate=True, past_key_values=self.past_key_values)
            
            self.output_ids = outputs.sequences
            self.past_key_values = outputs.past_key_values
            llm_outputs = self.tokenizer.batch_decode(self.output_ids, skip_special_tokens=False)[0].strip()
            print(llm_outputs, flush=True)
            self.action_seq = self.parse_actions(llm_outputs)
            print('actions', self.action_seq, flush=True)
            if len(self.action_seq) == 0: ## if generated llm without Specific values
                self.action_seq = [0]
        if self.step_id == 300:
            self.action_seq = [0] # stop for running too long time
        action = self.action_seq.pop(0)

        self.step_id += 1
        if self.step_id % self.num_frames == 0:
            self.policy.reset_for_env(0)
            self.output_ids = None
            self.past_key_values = None
            self.time_ids = []
        return [[action]]

    def step(self, obs):
        print('StreamvlnAgent step')
        start = time.time()
        action = self.inference(obs)
        end = time.time()
        print(f'总时间： {round(end-start,4)}s')
        return action
