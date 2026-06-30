#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO/LAS.h>
#include <CGAL/IO/read_las_points.h>
#include <CGAL/IO/read_points.h>
#include <CGAL/IO/OFF.h>
#include <CGAL/Real_timer.h>

#include "wasure.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <list>
#include <vector>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Vector_3 = K::Vector_3;
using Point_set = CGAL::Point_set_3<Point_3, Vector_3>;
using Wasure = CGAL::Wasure<Point_set>;

int main() {
  CGAL::Real_timer timer;
  timer.start();
  Point_set ps;

  std::ifstream ifile("C:/dev/sparkling-wasure/datas/lidar_hd_crop_1/LHD_FXX_0635_6857_PTS_C_LAMB93_IGN69.copc.crop.laz", std::ios::binary);
  //std::ifstream ifile("C:/dev/sparkling-wasure/datas/lidar_hd_crop_2/Semis_2021_0912_6457_LA93_IGN69.las", std::ios::binary);
  CGAL::IO::read_LAS(ifile, ps);

  //std::string filename = CGAL::data_file_path("points_3/building.ply");
  //std::string filename = CGAL::data_file_path("points_3/chair.xyz");
  //std::string filename = CGAL::data_file_path("points_3/cube_normals.xyz");
  //std::string filename = CGAL::data_file_path("points_3/sphere_1k.xyz");
//   bool res = CGAL::IO::read_points(filename, ps.index_back_inserter(),
//     CGAL::parameters::point_map(ps.point_push_map()).normal_map(ps.normal_push_map()));

  timer.stop();
  std::cout << "Reading " << ps.size() << " points: " << timer.time() << " seconds." << std::endl;
  timer.reset(); timer.start();

  Wasure wasure(ps);

  wasure.compute_features_svd();

  timer.stop();
  std::cout << "Computing PCAs: " << timer.time() << " seconds." << std::endl;
  timer.reset(); timer.start();

  std::size_t nov = wasure.adaptive_triangulation(0.05);

  timer.stop();
  std::cout << "Adaptive Delaunay triangulation " << timer.time() << " seconds. " << nov << " vertices in triangulation." << std::endl;
  timer.reset(); timer.start();

  wasure.compute_mass_function();

  timer.stop();
  std::cout << "Computing weights: " << timer.time() << " seconds." << std::endl;
  timer.reset(); timer.start();

  std::vector<Point_3> pts;
  std::vector<std::vector<std::uint32_t>> indices;

  std::vector<double> lambdas{ 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.73, 0.75, 0.77, 0.8, 0.9, 0.95, 0.99 };

  for (double l : lambdas) {
    pts.clear();
    indices.clear();
    std::cout << l << " ";
    wasure.extract_surface(l, std::back_inserter(pts), std::back_inserter(indices));
    if (pts.empty() || indices.empty())
      std::cout << l << " lambda led to empty result" << std::endl;
    CGAL::IO::write_OFF("wasure_" + std::to_string(l) + ".OFF", pts, indices);
  }

  return 0;
}
