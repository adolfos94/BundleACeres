#include <sensor.h>

#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfArray.h>

#include <sophus/se3.hpp>

const auto SKIP_N_FRAMES = 10;

std::ifstream SyntheticSensor::file_info_rgb;
std::ifstream SyntheticSensor::file_info_depth;
std::ifstream SyntheticSensor::file_info_poses;

std::vector<Sophus::SE3d> SyntheticSensor::load_all_poses()
{
	std::vector<Sophus::SE3d> poses;

	for (std::string pose; std::getline(file_info_poses, pose); )
	{
		int i = 0;
		// tx ty tz qx qy qz qw
		std::vector<double> extrinsics(7);
		std::string elem;
		std::stringstream iss(pose);
		while (iss >> elem)
		{
			if (i != 0)
			{
				extrinsics[i - 1] = stof(elem);
			}++i;
		}
		Sophus::Vector3d translation = Sophus::Vector3d(
			extrinsics[0], extrinsics[1], extrinsics[2]
		);

		Eigen::Quaterniond quaternion = Eigen::Quaterniond(
			extrinsics[6], extrinsics[3], extrinsics[4], extrinsics[5]
		);

		poses.push_back(Sophus::SE3d(quaternion, translation).inverse());
	}

	return poses;
}

SyntheticSensor::SyntheticSensor(const std::string& _dataset_dir)
	: current_frame_index{ 0 }, dataset_dir{ _dataset_dir }, image_width{ 640 }, image_height{ 480 } {
	const double fx = image_width * 4.1 / 4.54;
	const double fy = image_height * 4.1 / 3.42;
	const double cx = image_width / 2.0;
	const double cy = image_height / 2.0;

	intrinsics << fx, 0.0, cx,
		0.0, fy, cy,
		0.0, 0.0, 1.0;

	file_info_rgb = std::ifstream(dataset_dir + "rgb.txt");
	file_info_depth = std::ifstream(dataset_dir + "depth.txt");
	file_info_poses = std::ifstream(dataset_dir + "groundtruth.txt");

	if (!file_info_rgb.good() || !file_info_depth.good() || !file_info_poses.good())
	{
		std::cerr << "Error loading data.." << std::endl;
		return;
	}

	all_poses = load_all_poses();
}

FrameData SyntheticSensor::grab_frame() const {
	std::string rgb_path, depth_path;
	std::getline(file_info_rgb, rgb_path);
	std::getline(file_info_depth, depth_path);

	std::string filename_rgb, filename_depth;
	std::stringstream issr(rgb_path), issd(depth_path);
	while (issr >> filename_rgb);
	while (issd >> filename_depth);

	auto color = cv::imread(dataset_dir + filename_rgb, cv::IMREAD_COLOR);
	auto depth = cv::imread(dataset_dir + filename_depth, cv::IMREAD_UNCHANGED);

	if (color.empty() || depth.empty())
		throw std::runtime_error{ "Frames could not be grabbed" };

	//color.convertTo(color, CV_8UC3, 255.0);
	depth.convertTo(depth, CV_64FC1);

	FrameData frameData{ current_frame_index, color, depth, all_poses[current_frame_index] };
	current_frame_index++;

	return frameData;
}

bool SyntheticSensor::has_ended() const {
	return current_frame_index == all_poses.size();
}

Eigen::Matrix3d SyntheticSensor::get_intrinsics() {
	return intrinsics;
}