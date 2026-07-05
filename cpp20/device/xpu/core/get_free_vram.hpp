#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <zes_api.h>
#include <vector>
#include <cstdint>
#include <cstddef>

size_t get_free_vram(const sycl::device&);
