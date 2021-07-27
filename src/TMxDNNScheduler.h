#pragma once

#include "mxnet-cpp/lr_scheduler.h"


///Like FactorSheduler but with initial 
class TMxDNNScheduler : public mxnet::cpp::LRScheduler {
public:
	explicit TMxDNNScheduler(float base_lr, int step, float factor, float stop_factor_lr, int initial_num_update)
		: LRScheduler(base_lr) {
		count_ = 0;
		step_ = step;
		factor_ = factor;
		stop_factor_lr_ = stop_factor_lr;
		initial = initial_num_update;
	}

	float GetLR(unsigned num_update) override {
		while (num_update + initial > unsigned(count_ + step_)) {
			count_ += step_;
			base_lr_ *= factor_;
			if (base_lr_ < stop_factor_lr_) {
				base_lr_ = stop_factor_lr_;
				LG << "Update[" << num_update << "]: now learning rate arrived at " \
					<< base_lr_ << ", will not change in the future";
			}
			else {
				LG << "Update[" << num_update << "]: Change learning rate to " << base_lr_;
			}
		}
		return base_lr_;
	}

private:
	int count_;
	int initial;
	int step_;
	float factor_;
	float stop_factor_lr_;
};