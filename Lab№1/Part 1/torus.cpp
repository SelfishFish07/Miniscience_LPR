#include <set>
#include <gmsh.h>

int main(int argc, char **argv){
    gmsh::initialize();
    gmsh::model::add("torus");

    gmsh::model::occ::addTorus(0, 0, 0, 5, 2.5);

    gmsh::model::occ::synchronize();

    gmsh::option::setNumber("Mesh.MeshSizeMax", 1);
    
    gmsh::model::mesh::generate(2);

    gmsh::write("torus.msh");

    std::set<std::string> args(argv, argv + argc);
    if(!args.count("-nopopup")) gmsh::fltk::run();

    gmsh::finalize();

    return 0;
}