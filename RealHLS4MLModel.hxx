//Code generated automatically by TMVA for Inference of Model file [RealHLS4MLModel] at [Sat Mar 14 06:05:30 202] 

#ifndef ROOT_TMVA_SOFIE_REALHLS4MLMODEL
#define ROOT_TMVA_SOFIE_REALHLS4MLMODEL

#include <algorithm>
#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_RealHLS4MLModel{
namespace BLAS{
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
// initialized (weights and constant) tensors
std::vector<float> fTensor_dense_2_bias = std::vector<float>(5);
float * tensor_dense_2_bias = fTensor_dense_2_bias.data();
std::vector<float> fTensor_dense_2_kernel = std::vector<float>(160);
float * tensor_dense_2_kernel = fTensor_dense_2_kernel.data();
std::vector<float> fTensor_dense_1_bias = std::vector<float>(32);
float * tensor_dense_1_bias = fTensor_dense_1_bias.data();
std::vector<float> fTensor_dense_1_kernel = std::vector<float>(2048);
float * tensor_dense_1_kernel = fTensor_dense_1_kernel.data();
std::vector<float> fTensor_dense_bias = std::vector<float>(64);
float * tensor_dense_bias = fTensor_dense_bias.data();
std::vector<float> fTensor_dense_kernel = std::vector<float>(1024);
float * tensor_dense_kernel = fTensor_dense_kernel.data();

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(512);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor dense with size 256 bytes
float* tensor_dense = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor dense_relu with size 256 bytes
float* tensor_dense_relu = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 256);

 // Allocating memory for intermediate tensor dense_1 with size 128 bytes
float* tensor_dense_1 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 128);

 // Allocating memory for intermediate tensor dense_1_relu with size 128 bytes
float* tensor_dense_1_relu = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor dense_2 with size 20 bytes
float* tensor_dense_2 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 492);

 // Allocating memory for intermediate tensor dense_2_softmax with size 20 bytes
float* tensor_dense_2_softmax = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 472);


Session(std::string filename ="RealHLS4MLModel.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   using TMVA::Experimental::SOFIE::ReadTensorFromStream;
   ReadTensorFromStream(f, tensor_dense_2_bias, "tensor_dense_2_bias", 5);
   ReadTensorFromStream(f, tensor_dense_2_kernel, "tensor_dense_2_kernel", 160);
   ReadTensorFromStream(f, tensor_dense_1_bias, "tensor_dense_1_bias", 32);
   ReadTensorFromStream(f, tensor_dense_1_kernel, "tensor_dense_1_kernel", 2048);
   ReadTensorFromStream(f, tensor_dense_bias, "tensor_dense_bias", 64);
   ReadTensorFromStream(f, tensor_dense_kernel, "tensor_dense_kernel", 1024);
   f.close();

}

void doInfer(float const* tensor_input_layer,  std::vector<float> &output_tensor_dense_2_softmax ){


//--------- Gemm op_0 { 1 , 16 } * { 64 , 16 } -> { 1 , 64 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 64 * j;
      for (size_t k = 0; k < 64; k++) { 
         tensor_dense[y_index + k] = tensor_dense_bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_dense, true, false, 64, 1, 16, 1, tensor_dense_kernel, tensor_input_layer, 1,nullptr);

//------ RELU
   for (int id = 0; id < 64 ; id++){
      tensor_dense_relu[id] = ((tensor_dense[id] > 0 )? tensor_dense[id] : 0);
   }

//--------- Gemm op_2 { 1 , 64 } * { 32 , 64 } -> { 1 , 32 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 32 * j;
      for (size_t k = 0; k < 32; k++) { 
         tensor_dense_1[y_index + k] = tensor_dense_1_bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_dense_1, true, false, 32, 1, 64, 1, tensor_dense_1_kernel, tensor_dense_relu, 1,nullptr);

//------ RELU
   for (int id = 0; id < 32 ; id++){
      tensor_dense_1_relu[id] = ((tensor_dense_1[id] > 0 )? tensor_dense_1[id] : 0);
   }

//--------- Gemm op_4 { 1 , 32 } * { 5 , 32 } -> { 1 , 5 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 5 * j;
      for (size_t k = 0; k < 5; k++) { 
         tensor_dense_2[y_index + k] = tensor_dense_2_bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_dense_2, true, false, 5, 1, 32, 1, tensor_dense_2_kernel, tensor_dense_1_relu, 1,nullptr);

//------ RELU
   for (int id = 0; id < 5 ; id++){
      tensor_dense_2_softmax[id] = ((tensor_dense_2[id] > 0 )? tensor_dense_2[id] : 0);
   }
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_dense_2_softmax, output_tensor_dense_2_softmax, 5);
}



std::vector<float> infer(float const* tensor_input_layer){
   std::vector<float > output_tensor_dense_2_softmax;
   doInfer(tensor_input_layer, output_tensor_dense_2_softmax );
   return {output_tensor_dense_2_softmax};
}
};   // end of Session

} //TMVA_SOFIE_RealHLS4MLModel

#endif  // ROOT_TMVA_SOFIE_REALHLS4MLMODEL
