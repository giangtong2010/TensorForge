#pragma once
#include "Dtype.hpp"
#include "Device.hpp"
#include "_table.hpp"
#include <vector>
#include <cstddef>

namespace dispatcher {
    struct TypesAndBackends {
        Types type;
        Backends backend;
    };

    class Map {
        std::vector<Types> _map_dtype =
            std::vector<Types>(static_cast<size_t>(Types::count));

        std::vector<Backends> _map_device =
            std::vector<Backends>(static_cast<size_t>(Backends::count));

    public:
        void sign_up_dtype(cpp20::Dtype dtype, Types type) {
            _map_dtype[(size_t) dtype] = type;
        }
        void sign_up_device(cpp20::DeviceType dev_type, Backends backend) {
            _map_device[(size_t) dev_type] = backend;
        }

        TypesAndBackends get_type_and_backend(cpp20::Dtype dtype, cpp20::Device device) {
            auto dev_type = device._dev_type;
            Backends backend = _map_device[(size_t) dev_type];
            Types type = _map_dtype[(size_t) dtype];
            
            return TypesAndBackends {
                type,
                backend
            };
        }

        static Map& instance() {
            static Map map;
            return map;
        }
    };
}

#define MAP_DEVICE(device, backend)                                         \
namespace {                                                                 \
    static bool map_##device##_##backend = []() {                           \
        dispatcher::Map::instance().sign_up_device(                         \
            cpp20::DeviceType::device,                                      \
            dispatcher::Backends::backend                                   \
        );                                                                  \
        return true;                                                        \
    }();                                                                    \
}

#define MAP_DTYPE(dtype, type)                                              \
namespace {                                                                 \
    static bool map_##dtype##_##type = []() {                               \
        dispatcher::Map::instance().sign_up_dtype(                          \
            cpp20::Dtype::dtype,                                            \
            dispatcher::Types::type                                         \
        );                                                                  \
        return true;                                                        \
    }();                                                                    \
}
