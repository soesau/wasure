#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO/LAS.h>
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
using Point = K::Point_3;
using Vector = K::Vector_3;
using Point_set = CGAL::Point_set_3<Point, Vector>;
using Wasure = CGAL::Wasure<K, Point_set, Point_set::Point_map, Point_set::Vector_map>;

int main() {
  CGAL::Real_timer timer;
  timer.start();
  Point_set ps(true);
  std::string filename = CGAL::data_file_path("points_3/chair.xyz");
  //std::string filename = CGAL::data_file_path("points_3/chair_small.xyz");
  //std::string filename = CGAL::data_file_path("points_3/cube_normals.xyz");
  //std::string filename = CGAL::data_file_path("points_3/sphere_1k.xyz");
  bool res = CGAL::IO::read_points(filename, ps.index_back_inserter(),
    CGAL::parameters::point_map(ps.point_push_map()).normal_map(ps.normal_push_map()));

  timer.stop();
  std::cout << "Reading " << ps.size() << " points: " << timer.time() << " seconds." << std::endl;
  timer.reset(); timer.start();

  Wasure wasure(ps, ps.point_map(), ps.normal_map());

  wasure.compute_features();

  timer.stop();
  std::cout << "Computing PCAs: " << timer.time() << " seconds." << std::endl;
/*
  timer.reset(); timer.start();
  wasure.simplify(0.01);

  timer.stop();
  std::cout << "Simplify " << timer.time() << " seconds." << std::endl;*/
  timer.reset(); timer.start();

  wasure.compute_mass_function();

  timer.stop();
  std::cout << "Computing weights: " << timer.time() << " seconds." << std::endl;
  timer.reset(); timer.start();

  std::vector<Point> pts;
  std::vector<std::vector<std::uint32_t>> indices;

  std::vector<double> lambdas{ 0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.73, 0.75, 0.77, 0.8, 0.9, 0.95, 0.99 };

  for (double l : lambdas) {
    pts.clear();
    indices.clear();
    wasure.extract_surface(l, std::back_inserter(pts), std::back_inserter(indices));
    if (pts.empty() || indices.empty())
      std::cout << l << " lambda led to empty result" << std::endl;
    CGAL::IO::write_OFF("wasure_" + std::to_string(l) + ".OFF", pts, indices);
  }

  return 0;
}
