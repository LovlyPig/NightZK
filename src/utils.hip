#include "utils.h"

void check_gpu_ptr(void* ptr) {
    hipPointerAttribute_t attributes;
    hipError_t err = hipPointerGetAttributes(&attributes, ptr);

    if (err != hipSuccess) {
        printf("非法指针或未分配的内存: %s\n", hipGetErrorString(err));
    } else {
        if (attributes.type == hipMemoryTypeDevice) {
            printf("合法 GPU 指针，设备 ID: %d\n", attributes.device);
        } else if (attributes.type == hipMemoryTypeHost) {
            printf("这是主机 (CPU) 指针\n");
        }
    }
}
