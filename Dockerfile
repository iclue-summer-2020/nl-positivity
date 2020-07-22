FROM pypy:3.6-slim

WORKDIR /iclue-summer-2020/nl-positivity

RUN apt-get -y update    \
 && apt-get -y install   \
      bash               \
      build-essential    \
      git                \
      python3            \
      python3-dev        \
      python3-pip        \
      time               \

COPY . .

RUN pypy3 -m pip install --upgrade pip \
 && pypy3 -m pip install \
      cmake==3.17.3        \
      dill==0.3.2          \
      nlnum==1.2.1         \
      pathos==0.2.6
