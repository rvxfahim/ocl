#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include "C:/Program Files (x86)/AMD APP SDK/3.0/include/CL/cl.hpp"
#include <iostream>
int main()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	auto platform = platforms.front();
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	//print device sizes
    auto x=devices.size();
    std::cout << "Number of devices: " << x << std::endl;
    auto device = devices.front();
    std::cout << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    auto vendor = device.getInfo<CL_DEVICE_VENDOR>();
    std::cout << "Vendor: " << vendor << std::endl;
    auto version = device.getInfo<CL_DEVICE_VERSION>();
    std::cout << "Version: " << version << std::endl;
    auto driverVersion = device.getInfo<CL_DRIVER_VERSION>();
    std::cout << "Driver version: " << driverVersion << std::endl;
    auto maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    std::cout << "Max work group size: " << maxWorkGroupSize << std::endl;
    auto maxWorkItemDimensions = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
    std::cout << "Max work item dimensions: " << maxWorkItemDimensions << std::endl;
    auto maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    std::cout << "Max work item sizes: " << maxWorkItemSizes[0] << " " << maxWorkItemSizes[1] << " " << maxWorkItemSizes[2] << std::endl;
    //array a
    std::vector<int> a(10);
    for (int i = 0; i < 10; i++)
    {
        //randomly generate b
        a[i] = rand() % 10;
    }
    //array b
    std::vector<int> b(10);
    for (int i = 0; i < 10; i++)
    {
        //randomly generate b
        b[i] = rand() % 10;
    }
    //array c
    std::vector<int> c(10);
    for (int i = 0; i < 10; i++)
    {
        c[i] = 0;
    }
    //create context
    cl::Context context(devices);
    //create command queue
    cl::CommandQueue queue(context, device);
    //create buffers
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeof(int) * a.size());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY, sizeof(int) * b.size());
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(int) * c.size());
    //copy data to buffers
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(int) * a.size(), a.data());
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(int) * b.size(), b.data());
    //create program
    //kernel function to find difference of two arrays
    const char* source = R"CLC(
    __kernel void difference(__global const int *a, __global const int *b, __global int *c)
    {
        int i = get_global_id(0);
        c[i] = b[i] - a[i];
    }
    )CLC";
    cl::Program::Sources sources(1, std::make_pair(source, strlen(source)));
    cl::Program program(context, sources);
    //build program
    program.build(devices);
    //create kernel
    cl::Kernel kernel(program, "difference");
    //set kernel arguments
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    //create and execute command queue
    cl::NDRange global(a.size());
    cl::NDRange local(1);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
    //copy data from buffer to array
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(int) * c.size(), c.data());
    //print array c
    for (int i = 0; i < 10; i++)
    {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}