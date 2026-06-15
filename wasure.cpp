#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO/LAS.h>
#include <CGAL/IO/read_points.h>

#include "wasure.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <list>
#include <vector>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using DT = CGAL::Delaunay_triangulation_3<K>;
using Point = K::Point_3;
using Vector = K::Vector_3;
using Point_set = CGAL::Point_set_3<Point, Vector>;
using Wasure = CGAL::Wasure<K, Point_set, Point_set::Point_map, Point_set::Vector_map>;

int main() {
  Point_set ps(true);
  bool res = CGAL::IO::read_points(CGAL::data_file_path("points_3/chair.xyz"), ps.index_back_inserter(),
    CGAL::parameters::point_map(ps.point_push_map()).normal_map(ps.normal_push_map()));

  Wasure wasure(ps, ps.point_map(), ps.normal_map());

  wasure.compute_features();
  wasure.simplify(0.01);

  return 0;
}
