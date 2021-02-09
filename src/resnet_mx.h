#pragma once

#include "mxnet-cpp/MxNetCpp.h"

///in mxnet deconvolution is transpose convolution
mxnet::cpp::Symbol DeconvolutionNoBias(
	const std::string &symbol_name,
	mxnet::cpp::Symbol data,
	mxnet::cpp::Symbol weight,
	mxnet::cpp::Shape kernel,
	int num_filter,
	mxnet::cpp::Shape stride = mxnet::cpp::Shape(1, 1),
	mxnet::cpp::Shape dilate = mxnet::cpp::Shape(1, 1),
	mxnet::cpp::Shape pad = mxnet::cpp::Shape(0, 0),
	int num_group = 1,
	int64_t workspace = 512
){
	return mxnet::cpp::Operator("Deconvolution")
		.SetParam("kernel", kernel)
		.SetParam("num_filter", num_filter)
		.SetParam("stride", stride)
		.SetParam("dilate", dilate)
		.SetParam("pad", pad)
		.SetParam("num_group", num_group)
		.SetParam("workspace", workspace)
		.SetParam("no_bias", true)
		.SetInput("data", data)
		.SetInput("weight", weight)
		.CreateSymbol(symbol_name);
}

mxnet::cpp::Symbol ConvolutionNoBias(
	const std::string& symbol_name,
	mxnet::cpp::Symbol data,
	mxnet::cpp::Symbol weight,
	mxnet::cpp::Shape kernel,
	int num_filter,
	mxnet::cpp::Shape stride = mxnet::cpp::Shape(1, 1),
	mxnet::cpp::Shape dilate = mxnet::cpp::Shape(1, 1),
	mxnet::cpp::Shape pad = mxnet::cpp::Shape(0, 0),
	int num_group = 1,
	int64_t workspace = 512
) {
	return mxnet::cpp::Operator("Convolution")
		.SetParam("kernel", kernel)
		.SetParam("num_filter", num_filter)
		.SetParam("stride", stride)
		.SetParam("dilate", dilate)
		.SetParam("pad", pad)
		.SetParam("num_group", num_group)
		.SetParam("workspace", workspace)
		.SetParam("no_bias", true)
		.SetInput("data", data)
		.SetInput("weight", weight)
		.CreateSymbol(symbol_name);
}

mxnet::cpp::Symbol GetTrConv(
	const std::string &name,
	mxnet::cpp::Symbol data,
	int num_filter,
	mxnet::cpp::Shape kernel,
	mxnet::cpp::Shape stride,
	mxnet::cpp::Shape pad,
	bool with_relu,
	mx_float bn_momentum
) {
	mxnet::cpp::Symbol tr_conv_w(name + "_w");
	mxnet::cpp::Symbol tr_conv = DeconvolutionNoBias(name, data, tr_conv_w, kernel, num_filter, stride, mxnet::cpp::Shape(1, 1), pad, 1, 512);

	mxnet::cpp::Symbol gamma(name + "_gamma");
	mxnet::cpp::Symbol beta(name + "_beta");
	mxnet::cpp::Symbol mmean(name + "_mmean");
	mxnet::cpp::Symbol mvar(name + "_mvar");

	mxnet::cpp::Symbol bn = BatchNorm(name + "_bn", tr_conv, gamma, beta, mmean, mvar, 2e-5, bn_momentum, false);

	if (with_relu)
	{
		return Activation(name + "_relu", bn, "relu");
	}	else {
		return bn;
	}
}

mxnet::cpp::Symbol GetConv(
	const std::string& name, 
	mxnet::cpp::Symbol data,
	int  num_filter,
	mxnet::cpp::Shape kernel, 
	mxnet::cpp::Shape stride, 
	mxnet::cpp::Shape pad,
	bool with_relu,
	mx_float bn_momentum
) {
	mxnet::cpp::Symbol conv_w(name + "_w");
	mxnet::cpp::Symbol conv = ConvolutionNoBias(name, data, conv_w,	kernel, num_filter, stride, mxnet::cpp::Shape(1, 1), pad, 1, 512);

	mxnet::cpp::Symbol gamma(name + "_gamma");
	mxnet::cpp::Symbol beta(name + "_beta");
	mxnet::cpp::Symbol mmean(name + "_mmean");
	mxnet::cpp::Symbol mvar(name + "_mvar");

	mxnet::cpp::Symbol bn = BatchNorm(name + "_bn", conv, gamma, beta, mmean, mvar, 2e-5, bn_momentum, false);

	if (with_relu)
	{
		return Activation(name + "_relu", bn, "relu");
	}	else {
		return bn;
	}
}

mxnet::cpp::Symbol MakeTrBlock(
	const std::string &name,
	mxnet::cpp::Symbol data,
	int num_filter,
	bool dim_match,
	mx_float bn_momentum
) {
	mxnet::cpp::Shape stride;
	if (dim_match)
	{
		stride = mxnet::cpp::Shape(1, 1);
	}	else {
		stride = mxnet::cpp::Shape(2, 2);
	}

	mxnet::cpp::Symbol trconv1;
	if (dim_match)		//!!!’з как тут правильно сделать
	{
		trconv1 = GetTrConv(
			name + "_trconv1",
			data,
			num_filter,
			mxnet::cpp::Shape(3, 3),
			stride,
			mxnet::cpp::Shape(1, 1),
			true,
			bn_momentum
		);
	} else {
		auto trconv1p = GetTrConv(
			name + "_trconv1p",
			data,
			num_filter,
			mxnet::cpp::Shape(3, 3),
			mxnet::cpp::Shape(1, 1),
			mxnet::cpp::Shape(1, 1),
			false,
			bn_momentum
		);

		trconv1 = GetTrConv(
			name + "_trconv1",
			trconv1p,
			num_filter,
			mxnet::cpp::Shape(2, 2),
			stride,
			mxnet::cpp::Shape(0, 0),
			true,
			bn_momentum
		);
	}

	mxnet::cpp::Symbol trconv2 = GetTrConv(
			name + "_trconv2",
			trconv1,
			num_filter,
			mxnet::cpp::Shape(3, 3),
			mxnet::cpp::Shape(1, 1),
			mxnet::cpp::Shape(1, 1),
			false,
			bn_momentum
		);

	mxnet::cpp::Symbol shortcut;

	if (dim_match)
	{
		shortcut = data;
	}	else {
		mxnet::cpp::Symbol shortcut_w(name + "_proj_w");
		shortcut = DeconvolutionNoBias(
			name + "_proj",
			data,
			shortcut_w,
			mxnet::cpp::Shape(2, 2),
			num_filter,
			mxnet::cpp::Shape(2, 2),
			mxnet::cpp::Shape(1, 1),
			mxnet::cpp::Shape(0, 0),
			1,
			512
		);
	}

	mxnet::cpp::Symbol fused = shortcut + trconv2;
	return Activation(name + "_relu", fused, "relu");
}

mxnet::cpp::Symbol MakeBlock(
	const std::string &name,
	mxnet::cpp::Symbol data,
	int num_filter,
	bool dim_match,
	mx_float bn_momentum
) {
	mxnet::cpp::Shape stride;
	if (dim_match)
	{
		stride = mxnet::cpp::Shape(1, 1);
	}	else {
		stride = mxnet::cpp::Shape(2, 2);
	}

	mxnet::cpp::Symbol conv1 = GetConv(
		name + "_conv1", 
		data, 
		num_filter,	
		mxnet::cpp::Shape(3, 3), 
		stride, 
		mxnet::cpp::Shape(1, 1),
		true, 
		bn_momentum
	);

	mxnet::cpp::Symbol conv2 = GetConv(
		name + "_conv2", 
		conv1, 
		num_filter,
		mxnet::cpp::Shape(3, 3), 
		mxnet::cpp::Shape(1, 1), 
		mxnet::cpp::Shape(1, 1),
		false, 
		bn_momentum
	);

	mxnet::cpp::Symbol shortcut;

	if (dim_match)
	{
		shortcut = data;
	}	else {
		mxnet::cpp::Symbol shortcut_w(name + "_proj_w");
		shortcut = ConvolutionNoBias(
			name + "_proj", 
			data, 
			shortcut_w,
			mxnet::cpp::Shape(2, 2), 
			num_filter,
			mxnet::cpp::Shape(2, 2), 
			mxnet::cpp::Shape(1, 1), 
			mxnet::cpp::Shape(0, 0),
			1, 
			512
		);
	}

	mxnet::cpp::Symbol fused = shortcut + conv2;
	return Activation(name + "_relu", fused, "relu");
}

mxnet::cpp::Symbol getBody(
	mxnet::cpp::Symbol data,
	int num_level,
	int num_block,
	int num_filter,
	mx_float bn_momentum
) {
	for (int level = 0; level < num_level; level++) 
	{
		for (int block = 0; block < num_block; block++) 
		{
			data = MakeBlock(
				"level" + std::to_string(level + 1) + "_block" + std::to_string(block + 1),
				data, 
				num_filter * (std::pow(2, level)),
				(level == 0 || block > 0), 
				bn_momentum
			);
		}
	}
	return data;
}

mxnet::cpp::Symbol ResNetSymbol(
	int num_class,
	int num_level = 3,
	int num_block = 9,
	int num_filter = 16,
	mx_float bn_momentum = 0.9,
	mxnet::cpp::Shape pool_kernel = mxnet::cpp::Shape(8, 8)
) {
	// data and label
	mxnet::cpp::Symbol data = mxnet::cpp::Symbol::Variable("data");
	mxnet::cpp::Symbol data_label = mxnet::cpp::Symbol::Variable("label");

	mxnet::cpp::Symbol gamma("gamma");
	mxnet::cpp::Symbol beta("beta");
	mxnet::cpp::Symbol mmean("mmean");
	mxnet::cpp::Symbol mvar("mvar");

	mxnet::cpp::Symbol zscore = BatchNorm("zscore", data, gamma, beta, mmean, mvar, 0.001, bn_momentum);

	mxnet::cpp::Symbol conv = GetConv("conv0", zscore, num_filter, mxnet::cpp::Shape(3, 3), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(1, 1),	true, bn_momentum);

	mxnet::cpp::Symbol body = getBody(conv, num_level, num_block, num_filter, bn_momentum);

	mxnet::cpp::Symbol pool = Pooling("pool", body, pool_kernel, mxnet::cpp::PoolingPoolType::kAvg);

	mxnet::cpp::Symbol flat = Flatten("flatten", pool);

	mxnet::cpp::Symbol fc_w("fc_w"), fc_b("fc_b");
	mxnet::cpp::Symbol fc = mxnet::cpp::FullyConnected("fc", flat, fc_w, fc_b, num_class);

	return SoftmaxOutput("softmax", fc, data_label);
}