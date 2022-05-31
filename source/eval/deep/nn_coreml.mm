#include "nn_coreml.h"

#if defined(YANEURAOU_ENGINE_DEEP) && defined(COREML)

//#include "dlshogi_types.h"
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include "../../usi.h"

using namespace std;
using namespace Tools;


/// Model Prediction Input Type
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0)) __attribute__((visibility("hidden")))
@interface DlShogiResnetInput : NSObject<MLFeatureProvider>

/// input as 1 × 119 × 9 × 9 4-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * input;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithInput:(MLMultiArray *)input NS_DESIGNATED_INITIALIZER;

@end


/// Model Prediction Output Type
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0)) __attribute__((visibility("hidden")))
@interface DlShogiResnetOutput : NSObject<MLFeatureProvider>

/// output_policy as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * output_policy;

/// output_value as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * output_value;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithOutput_policy:(MLMultiArray *)output_policy output_value:(MLMultiArray *)output_value NS_DESIGNATED_INITIALIZER;

@end

@implementation DlShogiResnetInput

- (instancetype)initWithInput:(MLMultiArray *)input {
    self = [super init];
    if (self) {
        _input = input;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"input"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"input"]) {
        return [MLFeatureValue featureValueWithMultiArray:_input];
    }
    return nil;
}

@end

@implementation DlShogiResnetOutput

- (instancetype)initWithOutput_policy:(MLMultiArray *)output_policy output_value:(MLMultiArray *)output_value {
    self = [super init];
    if (self) {
        _output_policy = output_policy;
        _output_value = output_value;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"output_policy", @"output_value"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"output_policy"]) {
        return [MLFeatureValue featureValueWithMultiArray:_output_policy];
    }
    if ([featureName isEqualToString:@"output_value"]) {
        return [MLFeatureValue featureValueWithMultiArray:_output_value];
    }
    return nil;
}

@end

namespace Eval::dlshogi
{
	// モデルファイルの読み込み。
	Result NNCoreML::load(const std::string& model_filename , int gpu_id , int batch_size)
	{
		MLModelConfiguration* config = [MLModelConfiguration new];
		// 使用デバイス
		// MLComputeUnitsCPUOnly = 0,
		// MLComputeUnitsCPUAndGPU = 1,
		// MLComputeUnitsAll = 2
		// TODO: 選べるようにする。
		config.computeUnits = MLComputeUnitsAll;

		// このモデルをコンパイルし、mlmodelcを生成し、それをロードする
		NSString* model_path_ns = [NSString stringWithCString: model_filename.c_str() encoding:NSUTF8StringEncoding];
		NSString* modelc_path_ns = [NSString stringWithFormat: @"%@c", model_path_ns];

		
		NSFileManager *filemanager = [NSFileManager defaultManager];

		NSError *error = nil;
		if (![filemanager fileExistsAtPath: modelc_path_ns]) {
			if ([filemanager fileExistsAtPath: model_path_ns]) {
				sync_cout << "Compiling model" << sync_endl;
				NSURL* modelc_tmp_path = [MLModel compileModelAtURL: [NSURL fileURLWithPath: model_path_ns] error: &error];
				if (!modelc_tmp_path) {
					sync_cout << [[NSString stringWithFormat:@"info string Failed to compile model, %@", error] UTF8String] << sync_endl;
					exit(1);
				}
				BOOL ret = [filemanager moveItemAtURL: modelc_tmp_path toURL: [NSURL fileURLWithPath: modelc_path_ns] error: &error];
				if (!ret) {
					sync_cout << [[NSString stringWithFormat:@"info string Failed to move compiled model from %@ to %@, %@", modelc_tmp_path, modelc_path_ns, error] UTF8String] << sync_endl;
					exit(1);
				}
			} else {
				sync_cout << [[NSString stringWithFormat:@"info string Model %@ does not exist", model_path_ns] UTF8String] << sync_endl;
				exit(1);
			}
		} else {
			sync_cout << "info string Loading already compiled model" << sync_endl;
		}

		MLModel *model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath: modelc_path_ns] configuration:config error:&error];

		if (!model) {
			sync_cout << [[NSString stringWithFormat:@"info string Failed to load model, %@", error] UTF8String] << sync_endl;
			exit(1);
		}

		if (![model init]) {
			sync_cout << "info string Failed to initialize model" << sync_endl;
			exit(1);
		}

		this->model = model;

		input_buf = new float[NNCoreML::input_size * batch_size];

		return ResultCode::Ok;
	}

	// 使用可能なデバイス数を取得する。
	int NNCoreML::get_device_count() {
		return 1;
	}

	// NNによる推論
	void NNCoreML::forward(const int batch_size, PType* p1, PType* p2, NN_Input1* x1, NN_Input2* x2, NN_Output_Policy* y1, NN_Output_Value* y2)
	{
		NSError *error = nil;
		MLModel* model = reinterpret_cast<MLModel*>(this->model);

		// x1: [batch_size, 62, 9, 9], x2: [batch_size, 57, 9, 9]として与えられたものを、[batch_size, 119, 9, 9]に詰め替える
		for (int i = 0; i < batch_size; i++) {
			memcpy(&input_buf[119*9*9*i], &x1[i], 62 * 9 * 9 * sizeof(float));
			memcpy(&input_buf[119*9*9*i+62*9*9], &x2[i], 57 * 9 * 9 * sizeof(float));
		}

		MLMultiArray *model_input = [[MLMultiArray alloc] initWithDataPointer:input_buf shape:@[[NSNumber numberWithInt:batch_size], @119, @9, @9] dataType:MLMultiArrayDataTypeFloat32 strides:@[@(119*9*9), @(9*9), @9, @1] deallocator:NULL error:&error];
		if (error) {
			sync_cout << [[NSString stringWithFormat:@"info string CoreML inference array allocation failed, %@", error] UTF8String] << sync_endl;
			exit(1);
		}

		DlShogiResnetInput *input_ = [[DlShogiResnetInput alloc] initWithInput:model_input];
		id<MLFeatureProvider> outFeatures = [model predictionFromFeatures:input_ options:[[MLPredictionOptions alloc] init] error:&error];
		if (error) {
			sync_cout << [[NSString stringWithFormat:@"info string CoreML inference failed, %@", error] UTF8String] << sync_endl;
			exit(1);
		}

		DlShogiResnetOutput *model_output = [[DlShogiResnetOutput alloc] initWithOutput_policy:(MLMultiArray *)[outFeatures featureValueForName:@"output_policy"].multiArrayValue output_value:(MLMultiArray *)[outFeatures featureValueForName:@"output_value"].multiArrayValue];

		// 出力は動的確保された領域に書き出されるため、これを自前のバッファにコピー
		memcpy(y1, model_output.output_policy.dataPointer, batch_size * NNCoreML::policy_size * sizeof(float));
		memcpy(y2, model_output.output_value.dataPointer, batch_size * NNCoreML::value_size * sizeof(float));
	}

} // namespace Eval::dlshogi


#endif // defined(YANEURAOU_ENGINE_DEEP) && defined(COREML)

