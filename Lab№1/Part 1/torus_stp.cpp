#include <set>
#include <gmsh.h>

int main(int argc, char **argv){
    gmsh::initialize();
    gmsh::model::add("torus_stp");

    std::vector<std::pair<int, int> > v;
    gmsh::model::occ::importShapes("../torus.stp", v);

    gmsh::model::occ::synchronize();

    gmsh::option::setNumber("Mesh.MeshSizeMax", 1);
    gmsh::model::mesh::generate(2);

    gmsh::write("torus_stp.msh");

    std::set<std::string> args(argv, argv + argc);
    if(!args.count("-nopopup")) gmsh::fltk::run();

    gmsh::finalize();

    return 0;
}