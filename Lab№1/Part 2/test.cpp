#include <set>
#include <gmsh.h>
#include <vector>
#include <cmath>

int main(int argc, char **argv){
    gmsh::initialize();

    gmsh::model::add("test");
    gmsh::merge("../duck.stl");

    std::vector<std::pair<int, int> > s;
    gmsh::model::getEntities(s, 2);
    double angle = 40; 
    gmsh::model::mesh::classifySurfaces(angle*M_PI / 180., true, true);
    gmsh::model::mesh::createGeometry();

    gmsh::option::setNumber("Mesh.MeshSizeMax", 2.0); 

    gmsh::model::getEntities(s, 2);
    std::vector<int> sl;
    for(auto surf : s) sl.push_back(surf.second);
    int l = gmsh::model::geo::addSurfaceLoop(sl);
    gmsh::model::geo::addVolume({l});

    gmsh::model::geo::synchronize();

    gmsh::model::mesh::generate(3);

    gmsh::write("test.msh");

    std::set<std::string> args(argv, argv + argc);
    if(!args.count("-nopopup")) gmsh::fltk::run();

    gmsh::finalize();

    return 0;
}