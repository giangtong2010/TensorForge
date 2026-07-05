#include "get_free_vram.hpp"

size_t get_free_vram(const sycl::device& device) {
    auto ze_dev =
        sycl::get_native<
            sycl::backend::ext_oneapi_level_zero
        >(device);
    auto zes_dev = reinterpret_cast<zes_device_handle_t>(ze_dev);

    uint32_t mem_count = 0;
    zesDeviceEnumMemoryModules(zes_dev, &mem_count, nullptr);

    std::vector<zes_mem_handle_t> memories(mem_count);
    zesDeviceEnumMemoryModules(
        ze_dev,
        &mem_count,
        memories.data()
    );

    size_t free = 0;
    for (auto mem : memories) {
        zes_mem_state_t state {};
        state.stype = ZES_STRUCTURE_TYPE_MEM_STATE;

        zesMemoryGetState(mem, &state);
        free +=  state.free;
    }
    return free;
}
