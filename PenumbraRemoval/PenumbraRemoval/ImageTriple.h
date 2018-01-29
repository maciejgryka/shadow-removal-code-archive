#ifndef IMAGE_TRIPLE_H
#define IMAGE_TRIPLE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class ImageTriple {
public:
	ImageTriple() {};
	ImageTriple(
      const std::string& shad_path,
      const std::string& noshad_path,
      const std::string& mask_path,
      const std::string& maskp_path,
      const std::string& gmatte_path = ""):

		  shadow_path_(shad_path),
		  noshad_path_(noshad_path),
		  mask_path_(mask_path),
		  maskp_path_(maskp_path),
      gmatte_path_(gmatte_path) {
	}

	ImageTriple(
      const std::string& path,
      const std::string& shad_name,
      const std::string& noshad_name,
      const std::string& mask_name,
      const std::string& maskp_name,
      const std::string& gmatte_name=""):

		  shadow_path_(path + "\\" + shad_name),
		  noshad_path_(path +  "\\" + noshad_name),
		  mask_path_(path +  "\\" + mask_name),
		  maskp_path_(path +  "\\" + maskp_name),
		  gmatte_path_(path +  "\\" + gmatte_name)
	{}

 // ImageTriple(std::string path, std::string img_name, std::string noShadName, std::string maskName):
	//	shadow_path_(path + "\\" + shadName),
	//	noshad_path_(path +  "\\" + noShadName),
	//	mask_path_(path +  "\\" + maskName)
	//{}

	~ImageTriple(void) {};

	std::string getShadow() const {
		return shadow_path_;
	}

	std::string getNoShadow() const {
		return noshad_path_;
	}

	std::string getMask() const {
		return mask_path_;
	}

	std::string getMaskP() const {
		return maskp_path_;
	}

  std::string getGmatte() const {
		return gmatte_path_;
	}

private:
	std::string shadow_path_;
	std::string noshad_path_;
	std::string mask_path_;
	std::string maskp_path_;
	std::string gmatte_path_;
};
#endif //IMAGE_TRIPLE_H