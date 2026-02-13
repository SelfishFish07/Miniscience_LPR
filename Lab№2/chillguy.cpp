#include <iostream>
#include <cmath>
#include <vector>

#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkTetra.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>

#include <gmsh.h>

using namespace std;


class CalcNode
{
friend class CalcMesh;
protected:
    double x;
    double y;
    double z;

    double smth;

    double vx;
    double vy;
    double vz;
public:
    CalcNode() : x(0.0), y(0.0), z(0.0), smth(0.0), vx(0.0), vy(0.0), vz(0.0)
    {
    }

    CalcNode(double x, double y, double z, double smth, double vx, double vy, double vz) 
            : x(x), y(y), z(z), smth(smth), vx(vx), vy(vy), vz(vz)
    {
    }

    void move(double tau) {
        x += vx*tau;
        y += vy*tau;
        z += vz*tau;
    }
};

class Element
{
friend class CalcMesh;
protected:
    unsigned long nodesIds[4];
};

class CalcMesh
{
protected:
    vector<CalcNode> nodes;
    vector<Element> elements;
public:
    CalcMesh(const std::vector<double>& nodesCoords, const std::vector<std::size_t>& tetrsPoints){
        nodes.resize(nodesCoords.size() / 3);
        for (unsigned int i = 0; i < nodesCoords.size() / 3; i++) {
            double pointX = nodesCoords[i*3];
            double pointY = nodesCoords[i*3+1];
            double pointZ = nodesCoords[i*3+2];
            nodes[i] = CalcNode(pointX, pointY, pointZ, 0, 0.0, 0.0, 0.0);
        }

        elements.resize(tetrsPoints.size() / 4);
        for (unsigned int i = 0; i < tetrsPoints.size() / 4; i++){
            elements[i].nodesIds[0] = tetrsPoints[i*4] - 1;
            elements[i].nodesIds[1] = tetrsPoints[i*4+1] - 1;
            elements[i].nodesIds[2] = tetrsPoints[i*4+2] - 1;
            elements[i].nodesIds[3] = tetrsPoints[i*4+3] - 1;
        }
    }

    void doTimeStep(double tau, double currentTime){

        double amplitude = 1.0;
        double wavelength = 50;
        double k = 2 * M_PI / wavelength;
        double w = 20.0;

        double Cx = nodes[2151].x;
        double Cy = nodes[2151].y;
        double Cz = nodes[2151].z;

        //Bottom points of ears
        auto LEarB = nodes[27790];
        auto REarB = nodes[42759];

        double v_ampltitude = 3;
        double ear_frequency = 40.0;

        for (unsigned int i = 0; i < nodes.size(); i++){
            nodes[i].vx = 0;
            nodes[i].vy = 0;
            nodes[i].vz = 0;

            double r = sqrt(pow(nodes[i].x - Cx,2) + pow(nodes[i].y - Cy,2) + pow(nodes[i].z - Cz,2));
            nodes[i].smth = amplitude * sin(k * r - w * currentTime) * exp (-0.03 * r);
            
            if  (nodes[i].z>LEarB.z-1 && nodes[i].y<5 && nodes[i].x>18){
                double dr = sqrt(pow(nodes[i].x - LEarB.x,2) + pow(nodes[i].y - LEarB.y,2) + pow(nodes[i].z - LEarB.z,2));
                nodes[i].vy = dr * v_ampltitude * sin(ear_frequency*currentTime);
            } else if (nodes[i].z>REarB.z-1 && nodes[i].y>5 && nodes[i].x>18) {
                double dr = sqrt(pow(nodes[i].x - REarB.x,2) + pow(nodes[i].y - REarB.y,2) + pow(nodes[i].z - REarB.z,2));
                nodes[i].vy = -1*dr * v_ampltitude * sin(ear_frequency*currentTime);
            }
            nodes[i].move(tau);
        }
    }

    void snapshot(unsigned int snap_number){
        vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
        vtkSmartPointer<vtkPoints> dumpPoints = vtkSmartPointer<vtkPoints>::New();

        auto smth = vtkSmartPointer<vtkDoubleArray>::New();
        smth->SetName("smth");

        auto vel = vtkSmartPointer<vtkDoubleArray>::New();
        vel->SetName("vel");
        vel->SetNumberOfComponents(3);

        for (unsigned int i = 0; i < nodes.size(); i++){
            dumpPoints->InsertNextPoint(nodes[i].x,nodes[i].y,nodes[i].z);

            double _vel[3] = {nodes[i].vx, nodes[i].vy, nodes[i].vz};
            vel->InsertNextTuple(_vel);
            smth->InsertNextValue(nodes[i].smth);
        }


        unstructuredGrid->SetPoints(dumpPoints);
        unstructuredGrid->GetPointData()->AddArray(vel);
        unstructuredGrid->GetPointData()->AddArray(smth);

        for (unsigned int i = 0; i < elements.size(); i++){
            auto tetra = vtkSmartPointer<vtkTetra>::New();
            tetra->GetPointIds()->SetId(0, elements[i].nodesIds[0]);
            tetra->GetPointIds()->SetId(1, elements[i].nodesIds[1]);
            tetra->GetPointIds()->SetId(2, elements[i].nodesIds[2]);
            tetra->GetPointIds()->SetId(3, elements[i].nodesIds[3]);
            unstructuredGrid->InsertNextCell(tetra->GetCellType(), tetra->GetPointIds());
        }
        
        string filename = "./chill_paraview/chill_step_" + std::to_string(snap_number) + ".vtu";
        vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        writer->SetFileName(filename.c_str());
        writer->SetInputData(unstructuredGrid);
        writer->Write();


    }
};

int main()
{
    double tau = 0.01;

    const unsigned int GMSH_TETR_CODE = 4;

    gmsh::initialize();

    gmsh::model::add("test");
    gmsh::merge("../chillguy.stl");

    
    double angle = 60; 
    gmsh::model::mesh::classifySurfaces(angle * M_PI / 180., true, true);
    gmsh::model::mesh::createGeometry();

    std::vector<std::pair<int, int> > s;
    gmsh::model::getEntities(s, 2);
    std::vector<int> sl;
    for(auto surf : s) sl.push_back(surf.second);
    int l = gmsh::model::geo::addSurfaceLoop(sl);
    gmsh::model::geo::addVolume({l});

    gmsh::model::geo::synchronize();

    int f = gmsh::model::mesh::field::add("MathEval");
    gmsh::model::mesh::field::setString(f, "F", "1");
    gmsh::model::mesh::field::setAsBackgroundMesh(f);

    gmsh::model::mesh::generate(3);

    std::vector<double> nodesCoord;
    std::vector<std::size_t> nodeTags;
    std::vector<double> parametricCoord;
    gmsh::model::mesh::getNodes(nodeTags, nodesCoord, parametricCoord);

    std::vector<std::size_t>* tetrsNodesTags = nullptr;
    std::vector<int> elementTypes;
    std::vector<std::vector<std::size_t>> elementTags;
    std::vector<std::vector<std::size_t>> elementNodeTags;
    gmsh::model::mesh::getElements(elementTypes, elementTags, elementNodeTags);
    for(unsigned int i = 0; i < elementTypes.size(); i++) {
        if(elementTypes[i] != GMSH_TETR_CODE)
            continue;
        tetrsNodesTags = &elementNodeTags[i];
    }

    cout << "The model has " <<  nodeTags.size() << " nodes and " << tetrsNodesTags->size() / 4 << " tetrs." << endl;
    for(int i = 0; i < nodeTags.size(); ++i) {
        assert(i == nodeTags[i] - 1);
    }
    assert(tetrsNodesTags->size() % 4 == 0);

    CalcMesh mesh(nodesCoord, *tetrsNodesTags);

    gmsh::finalize();

    mesh.snapshot(0);

    double currentTime = 0.0;
    for(unsigned int step = 1; step < 70; step++) {
        mesh.doTimeStep(tau, currentTime);
        mesh.snapshot(step);
        currentTime += tau;
    }

    return 0;


}
