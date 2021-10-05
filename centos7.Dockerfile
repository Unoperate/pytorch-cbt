# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A lot of the code here is "borrowed" from the
# https://github.com/googleapis/google-cloud-cpp project.

FROM quay.io/pypa/manylinux2014_x86_64
ARG NCPU=4

# ```bash
RUN yum install -y centos-release-scl yum-utils
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms
RUN yum makecache && \
    yum install -y \
	automake \
	ccache \
	cmake3 \
	curl-devel \
	devtoolset-7 \
	gawk \
	gcc \
	gcc-c++ \
    git \
    less \
    m4 \
	make \
	ninja-build \
	openssl-devel \
	patch \
    python3-devel \
	tar \
	unzip \
	vim \
	wget \
	which \
	zip \
	zlib-devel


RUN ln -sf /usr/bin/cmake3 /usr/bin/cmake && ln -sf /usr/bin/ctest3 /usr/bin/ctest
# ```

# CentOS-7 ships with `pkg-config` 0.27.1, which has a
# [bug](https://bugs.freedesktop.org/show_bug.cgi?id=54716) that can make
# invocations take extremely long to complete. If you plan to use `pkg-config`
# with any of the installed artifacts, you'll want to upgrade it to something
# newer. If not, `yum install pkgconfig` should work instead.

# ```bash
WORKDIR /var/tmp/build/pkg-config-cpp
RUN curl -sSL https://pkgconfig.freedesktop.org/releases/pkg-config-0.29.2.tar.gz | \
    tar -xzf - --strip-components=1 && \
    ./configure --with-internal-glib && \
    make -j ${NCPU:-4} && \
    make install && \
    ldconfig
# ```


# The following steps will install libraries and tools in `/usr/local`. By
# default CentOS-7 does not search for shared libraries in these directories,
# there are multiple ways to solve this problem, the following steps are one
# solution:

# ```bash
RUN (echo "/usr/local/lib" ; echo "/usr/local/lib64") | \
    tee /etc/ld.so.conf.d/usrlocal.conf
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig:/usr/lib64/pkgconfig
ENV PATH=/usr/local/bin:${PATH}
# ```

# Install the Cloud SDK and some of the emulators. We use the emulators to run
# integration tests.
COPY install-cloud-sdk.sh /var/tmp/ci/
WORKDIR /var/tmp/downloads
RUN /var/tmp/ci/install-cloud-sdk.sh
ENV CLOUD_SDK_LOCATION=/usr/local/google-cloud-sdk
ENV PATH=${CLOUD_SDK_LOCATION}/bin:${PATH}

# pyTorch is compiled using the new libstdc++ ABI. When C++11 was introduced,
# libstdc++ maintainers had to rewrite parts of the library. In order not to
# break binary compatibility with previously compiled programs, they decided
# that new, C++11-conformant library implementation will be in an inline
# namespace (__cxx11), which means that symbols for that implementation are
# different (contain the __cxx11 namespace name in them). They also allowed the
# users to chnage this behavior (i.e. break binary compatibility but link
# against symbols without __cxx11 substring in them) if they set the flag
# `_GLIBCXX_USE_CXX11_ABI` to `0`. Ubuntu (and Fedora) compile libraries with
# this definition set to `1`. pyTorch, however, is compiled with it set to `0`,
# which means that our C++ Bigtable plugin and all its dependencies have to be
# compiled that way too.

# In order to make it easy for the users, we compile the plugin and its
# dependecies statically. This is especially helpful given the
# `_GLIBCXX_USE_CXX11_ABI` incompatibility. However, the end result is a library
# loaded by python, so we need to compile the code as position independent
# (hence CMAKE_POSITION_INDEPENDENT_CODE is set to ON).

# #### c-ares

# Recent versions of gRPC require c-ares >= 1.11, while CentOS-7
# distributes c-ares-1.10. Manually install a newer version:

WORKDIR /var/tmp/build/c-ares
RUN curl -sSL https://github.com/c-ares/c-ares/archive/cares-1_14_0.tar.gz | \
    tar -xzf - --strip-components=1 && \
    ./buildconf && \
    CXXFLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' \ 
    ./configure --with-pic --disable-shared && \
    make -j ${NCPU:-4} && \
    make install && \
    ldconfig

# libre2 is required by gRPC
WORKDIR /var/tmp/build/re2
RUN curl -sSL https://github.com/google/re2/archive/refs/tags/2021-08-01.tar.gz | \
    tar -xzf - --strip-components=1 && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=NO \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCRC32C_BUILD_TESTS=OFF \
        -DCRC32C_BUILD_BENCHMARKS=OFF \
        -DCRC32C_USE_GLOG=OFF \
        -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' \
        -H. -Bcmake-out && \
    cmake --build cmake-out -- -j ${NCPU:-4} && \
    cmake --build cmake-out --target install -- -j ${NCPU:-4} && \
    ldconfig

# Abseil
WORKDIR /var/tmp/build/abseil-cpp
RUN curl -sSL https://github.com/abseil/abseil-cpp/archive/20210324.2.tar.gz | \
    tar -xzf - --strip-components=1 && \
    sed -i 's/^#define ABSL_OPTION_USE_\(.*\) 2/#define ABSL_OPTION_USE_\1 0/' "absl/base/options.h" && \
    CXXFLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DBUILD_SHARED_LIBS=NO \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_CXX_STANDARD=11 \
        -H. -Bcmake-out && \
    cmake --build cmake-out -- -j ${NCPU:-4} && \
    cmake --build cmake-out --target install -- -j ${NCPU:-4} && \
    ldconfig

# Protobuf

# We need to install a version of Protobuf that is recent enough to support the
# Google Cloud Platform proto files:
WORKDIR /var/tmp/build/protobuf
RUN curl -sSL https://github.com/protocolbuffers/protobuf/archive/v3.17.3.tar.gz | \
    tar -xzf - --strip-components=1 && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=no \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -Dprotobuf_BUILD_TESTS=OFF \
        -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' \
        -Hcmake -Bcmake-out && \
    cmake --build cmake-out -- -j ${NCPU:-4} && \
    cmake --build cmake-out --target install -- -j ${NCPU:-4} && \
    ldconfig

# gRPC
WORKDIR /var/tmp/build/grpc
RUN curl -sSL https://github.com/grpc/grpc/archive/v1.39.0.tar.gz | \
    tar -xzf - --strip-components=1 && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DgRPC_ABSL_PROVIDER=package \
        -DgRPC_CARES_PROVIDER=package \
        -DgRPC_PROTOBUF_PROVIDER=package \
        -DgRPC_RE2_PROVIDER=package \
        -DgRPC_SSL_PROVIDER=package \
        -DgRPC_ZLIB_PROVIDER=package \
        -DBUILD_SHARED_LIBS=no \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' \
        -H. -Bcmake-out && \
    cmake --build cmake-out -- -j ${NCPU:-4} && \
    cmake --build cmake-out --target install -- -j ${NCPU:-4} && \
    ldconfig

# crc32c
WORKDIR /var/tmp/build/crc32c
RUN curl -sSL https://github.com/google/crc32c/archive/1.1.0.tar.gz | \
    tar -xzf - --strip-components=1 && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=NO \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCRC32C_BUILD_TESTS=OFF \
        -DCRC32C_BUILD_BENCHMARKS=OFF \
        -DCRC32C_USE_GLOG=OFF \
        -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' \
        -H. -Bcmake-out && \
    cmake --build cmake-out -- -j ${NCPU:-4} && \
    cmake --build cmake-out --target install -- -j ${NCPU:-4} && \
    ldconfig

# nlohmann_json
WORKDIR /var/tmp/build/json
RUN curl -sSL https://github.com/nlohmann/json/archive/v3.9.1.tar.gz | \
    tar -xzf - --strip-components=1 && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=no \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_TESTING=OFF \
        -H. -Bcmake-out/nlohmann/json && \
    cmake --build cmake-out/nlohmann/json --target install -- -j ${NCPU:-4} && \
    ldconfig

# google-cloud-cpp
WORKDIR /var/tmp/build/google-cpp
RUN curl -sSL \
  https://github.com/googleapis/google-cloud-cpp/archive/refs/tags/v1.30.1.tar.gz  | \
  tar xfvz - && \
  cd google-cloud-cpp-1.30.1 && \
  cmake -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DGOOGLE_CLOUD_CPP_ENABLE_EXAMPLES=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' \
        -H. -B cmake-out && \
  cmake --build cmake-out -j ${NCPU:-4} && \
  cmake --install cmake-out --component google_cloud_cpp_development && \
  ldconfig

WORKDIR /opt/pytorch
