#pragma once
#include <sycl/sycl.hpp>
#include <ze_api.h>
#include <zes_api.h>
#include <vector>
#include <cstdint>
#include <cstddef>

size_t get_free_vram(const sycl::device&);
