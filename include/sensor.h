#pragma once

#include <types.h>


constexpr auto SKIP_N_FRAMES = 10;

class SyntheticSensor {
public:

	SyntheticSensor(const std::string& _dataset_dir);

	std::vector<Sophus::SE3d> load_all_poses();
	FrameData grab_frame() const;
	bool has_ended() const;

	Eigen::Matrix3d get_intrinsics();

private:
	const std::string dataset_dir;
	const int image_width, image_height;
	Eigen::Matrix3d intrinsics;

	mutable size_t current_frame_index;

	std::vector<Sophus::SE3d> all_poses;

	static std::ifstream file_info_rgb;
	static std::ifstream file_info_depth;
	static std::ifstream file_info_poses;
};
