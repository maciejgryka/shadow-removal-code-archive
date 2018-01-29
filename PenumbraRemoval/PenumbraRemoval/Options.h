#ifndef OPTIONS_H
#define OPTIONS_H

#include <iostream>
#include <string>
#include <map>
#include <utility>

class Options {
public:

  Options() {};
  ~Options() {};

  bool IsParamValid(const std::string& key) const {
    return option_map_.count(key) == 1;
  }

  int GetParamInt(const std::string& key) const {
    CheckParamExists(key);
		return atoi(option_map_.at(key).c_str());
	};

	float GetParamFloat(const std::string& key) const {
    CheckParamExists(key);
		return float(atof(option_map_.at(key).c_str()));
	};

	std::string GetParamString(const std::string& key) const {
    CheckParamExists(key);
		return option_map_.at(key);
	};

  bool GetParamBool(const std::string& key) const {
    CheckParamExists(key);
    return option_map_.at(key).compare(std::string("0")) != 0;
  }

  std::map<std::string, bool> GetActiveFeatures() const {
    std::map<std::string, bool> activeFeatures;
    activeFeatures["feature_intensity"]             = GetParamInt("feature_intensity") == 1;
    activeFeatures["feature_gradient_orientation"]  = GetParamInt("feature_gradient_orientation") == 1;
    activeFeatures["feature_gradient_magnitude"]    = GetParamInt("feature_gradient_magnitude") == 1;
    activeFeatures["feature_gradient_xy"]           = GetParamInt("feature_gradient_xy") == 1;
    activeFeatures["feature_distance_transform"]    = GetParamInt("feature_distance_transform") == 1;
    activeFeatures["feature_polar_angle"]           = GetParamInt("feature_polar_angle") == 1;
    activeFeatures["feature_gmatte"]                = GetParamInt("feature_gmatte") == 1;
    
    return activeFeatures;
  };

  void SetParam(const std::string key, const std::string& value) {
    option_map_[key] = value;
  }

  void SetParam(const std::pair<std::string, std::string>& key_value) {
    option_map_[key_value.first] = key_value.second;
  }

private:
  void CheckParamExists(const std::string& param) const {
    if (option_map_.count(param) == 0) {
      std::cerr << "ERROR: param '" << param << "' not found in options." << std::endl;
    }
  }

	std::map<std::string, std::string> option_map_;
};

#endif // OPTIONS_H