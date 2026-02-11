#include <gmsh.h>
#include <vector>
#include <utility>
#include <set>

int main(int argc, char **argv){
    gmsh::initialize();
    gmsh::model::add("torus");

    gmsh::model::occ::addTorus(0, 0, 0, 5, 2, 1);
    gmsh::model::occ::addTorus(0, 0, 0, 5, 1, 2);

    std::vector<std::pair<int, int> > ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    gmsh::model::occ::cut({{3, 1}}, {{3, 2}}, ov, ovv, 3);


    gmsh::model::occ::synchronize();

    gmsh::option::setNumber("Mesh.MeshSizeMax", 0.5);
    
    gmsh::model::mesh::generate(3);

    gmsh::write("torus.msh");

    std::set<std::string> args(argv, argv + argc);
    if(!args.count("-nopopup")) gmsh::fltk::run();

    gmsh::finalize();

    return 0;
}