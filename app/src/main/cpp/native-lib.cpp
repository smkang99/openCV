
#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS /*let's give a chance for OpenCL 1.1 devices*/
#include <jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <GLES2/gl2.h>
#include <CL/cl.hpp>
#include "common.hpp"


using namespace cv;

extern "C" JNIEXPORT void

JNICALL
Java_net_jonreynolds_androidopencvcamera_MyGLSurfaceView_processFrame(JNIEnv *env, jobject /* this */,
                                                                      jint texIn, jint texOut,
                                                                      jint w, jint h,
                                                                      jboolean frontFacing) {
    static UMat m;

    LOGD("Processing on CPU");
    int64_t t;
    m.create(h, w, CV_8UC4);

    UMat canvas, canvasPart, rimg, rmap[2];
    double sf = 600./MAX(w, h);
    int width = cvRound(w*sf);
    int height = cvRound(h*sf);
    canvas.create(height, width*2, CV_8UC3);
    Size imageSizeRGB;
    // read
    t = getTimeMs();
    // expecting FBO to be bound, read pixels to mat
    glReadPixels(0, 0, m.cols, m.rows, GL_RGBA, GL_UNSIGNED_BYTE, m.getMat(ACCESS_WRITE).data);
    LOGD("glReadPixels() costs %d ms", getTimeInterval(t));

    t = getTimeMs();
    // Check if we should flip image due to frontFacing
    // I don't think this should be required, but I can't find
    // a way to get the OpenCV Android SDK to do this properly
    // (also, time taken to flip image is negligible)
    if(frontFacing){
        flip(m, m, 1);
    }
    LOGD("flip() costs %d ms", getTimeInterval(t));

    imageSizeRGB = m.size();
    Scalar_<double> data[] = {0.3958825574552909, 3.648558289601505, -6390.940415385947
    , -0.6101491284401678, 3.647061849359864, -4378.900006059477,
                            0.0001920079015145303, 0.001769595572088184, -2.827202996448164};
    //UMat H2= UMat(3, 3, CV_32F,  data, USAGE_DEFAULT);

    Mat H2 = Mat::zeros(3, 3, CV_64FC1);
    // modify
    t = getTimeMs();
    //cvtColor(m, m, CV_BGRA2GRAY);
    ////Laplacian(m, m, CV_8U);
    ////multiply(m, 10, m);
    ////remap(m, rimg, rmap[0], rmap[1], INTER_LINEAR);
    //warpPerspective(m, rimg, H2, imageSizeRGB);
    //cvtColor(m, m, CV_GRAY2BGRA);
    LOGD("*** OpenCL info ***");

#if 0
    if (!ocl::haveOpenCL())
    {
        LOGD("OpenCL is not available...");
        return;
    }

    ocl::Context context;
    if (!context.create(ocl::Device::TYPE_GPU))
    {
        LOGD("Failed creating the context...");
        return;
    }

    // In OpenCV 3.0.0 beta, only a single device is detected.
    std::cout << context.ndevices() << " GPU devices are detected." << std::endl;
    for (int i = 0; i < context.ndevices(); i++)
    {
        cv::ocl::Device device = context.device(i);
        std::cout << "name                 : " << device.name() << std::endl;
        std::cout << "available            : " << device.available() << std::endl;
        std::cout << "imageSupport         : " << device.imageSupport() << std::endl;
        std::cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << std::endl;
        std::cout << std::endl;
    }

    // Select the first device
    cv::ocl::Device(context.device(0));
#endif
    LOGD("Laplacian() costs %d ms", getTimeInterval(t));

    // write back
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texOut);
    t = getTimeMs();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m.cols, m.rows, GL_RGBA, GL_UNSIGNED_BYTE, m.getMat(ACCESS_READ).data);
    LOGD("glTexSubImage2D() costs %d ms", getTimeInterval(t));
}