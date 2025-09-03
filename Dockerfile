FROM crpi-mdum1jboc8276vb5.cn-beijing.personal.cr.aliyuncs.com/iros-challenge/internnav:v1.0

ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai

WORKDIR /root

COPY . /root/InternNav

WORKDIR /root/InternNav

