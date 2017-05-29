//----------------------------------------------------------------------------------------------
//	Filename:	nbody.cpp
//	Author:		Keith Bugeja
//----------------------------------------------------------------------------------------------
//  CPS3227 assignment for academic year 2017/2018:
//	Sample naive [O(n^2)] implementation for the N-Body problem.
//----------------------------------------------------------------------------------------------

#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <vector>
#include <time.h>
using namespace std;

#include "vector2.h"


#include <unistd.h>


/*
 * Constant definitions for field dimensions, and particle masses
 */

const int fieldWidth = 1000;
const int fieldHalfWidth = fieldWidth >> 1;
const int fieldHeight = 1000;
const int fieldHalfHeight = fieldHeight >> 1;

const float minBodyMass = 2.5f;
const float maxBodyMassVariance = 5.f;
#define	NUM_THREADS	12
//switch to control parallelization - if set to 1 openmp directives will be included
#define	DO_PARALLELIZE	0

/*
 * Particle structure
 */
struct Particle
{
	Vector2 Position;
	Vector2 Velocity;
	float	Mass;
	
	Particle(void) 
		: Position( ((float)rand()) / RAND_MAX * fieldWidth - fieldHalfWidth,
					((float)rand()) / RAND_MAX * fieldHeight - fieldHalfHeight)
		, Velocity( 0.f, 0.f )
		, Mass ( ((float)rand()) / RAND_MAX * maxBodyMassVariance + minBodyMass )
	{ }
};



/*
 * read inital values for particles from input file
 */
void readInitialBodyValuesFromFile(std::vector<Particle> &p_bodies, const char* filename)
{
    std::ifstream  data(filename);

    std::string line;
    while(std::getline(data,line))
    {
        std::stringstream  lineStream(line);
        std::string        cell;
        std::string         temp;

        Particle particle;

        std::getline(lineStream,cell,',');
        particle.Mass = ::atof(cell.c_str());
        std::getline(lineStream,cell,',');
        particle.Position.X = ::atof(cell.c_str());
        std::getline(lineStream,cell,',');

        particle.Position.Y = ::atof(cell.c_str());
        particle.Velocity.X = 0.0f;
        particle.Velocity.Y = 0.0f;
        p_bodies.push_back(particle);

        //cout << "mass: " << particle.Mass <<  "    X: " << particle.Position.X <<  "    Y: " << particle.Position.Y <<  "\n";
    }
}

/*
 * Compute forces of particles exerted on one another
 */
void ComputeForces(std::vector<Particle> &p_bodies, float p_gravitationalTerm, float p_deltaT)
{
    Vector2 direction,
            force, acceleration;

	float distance;

    omp_set_num_threads(NUM_THREADS);
    int numOfThreads = omp_get_num_threads();

    size_t j = 0;
    size_t k = 0;

    #if DO_PARALLELIZE == 1
        #pragma omp declare reduction(plusVector2 : Vector2 : omp_out = omp_out + omp_in) \
                 initializer (omp_priv= 0.f)
    #endif

    force = 0.f, acceleration = 0.f;

    #if DO_PARALLELIZE == 1
        #pragma omp parallel for schedule(static) private(j,acceleration) shared(p_gravitationalTerm) num_threads(NUM_THREADS)
    #endif
    for (j = 0; j < p_bodies.size(); ++j)
    {
        Particle &p1 = p_bodies[j];

        #if DO_PARALLELIZE == 1
            #pragma omp parallel for schedule(static) private(k,direction,distance) reduction(plusVector2: force) num_threads(NUM_THREADS)
        #endif
        for (k = 0; k < p_bodies.size(); ++k) {
            if (k == j) continue;

            Particle &p2 = p_bodies[k];

            // Compute direction vector
            direction = p2.Position - p1.Position;

            // Limit distance term to avoid singularities
            distance = std::max<float>(0.5f * (p2.Mass + p1.Mass), direction.Length());

            // Accumulate force
            force += direction / (distance * distance * distance) * p2.Mass;
        }


        acceleration = force * p_gravitationalTerm;

        // Integrate velocity (m/s)
        p1.Velocity += acceleration * p_deltaT;
        //reset acceleration and force values
        force = 0.f, acceleration = 0.f;

    }

}

/*
 * Update particle positions
 */
void MoveBodies(std::vector<Particle> &p_bodies, float p_deltaT)
{
    omp_set_num_threads(NUM_THREADS);
    size_t j;
    int p_bodies_size = p_bodies.size();
    //static scheduling was done because of know number
    #if DO_PARALLELIZE == 1
        #pragma omp parallel for shared(p_bodies,p_deltaT,p_bodies_size) private(j) schedule(static) num_threads(NUM_THREADS)
    #endif
	for (j = 0; j < p_bodies_size; ++j)
	{
		p_bodies[j].Position += p_bodies[j].Velocity * p_deltaT;
	}
}

/*
 * Commit particle masses and positions to file in CSV format
 */
void PersistPositions(const std::string &p_strFilename, std::vector<Particle> &p_bodies)
{
	std::cout << "\nWriting to file: " << p_strFilename << "\n";
	std::ofstream output(p_strFilename.c_str());
	
	if (output.is_open())
	{	
		for (int j = 0; j < p_bodies.size(); j++)
		{
			output << 	p_bodies[j].Mass << ", " <<
				p_bodies[j].Position.Element[0] << ", " <<
				p_bodies[j].Position.Element[1] << "\n";
		}
		
		output.close();
	}
	else
		std::cerr << "Unable to persist data to file:" << p_strFilename << "\n";

}

int main(int argc, char **argv)
{
	int particleCount = 0;
	const int maxIteration = 1000;
	const float deltaT = 0.01f;
	const float gTerm = 20.f;

	std::stringstream fileOutput;
	std::vector<Particle> bodies;

	//for (int bodyIndex = 0; bodyIndex < particleCount; ++bodyIndex)
	//	bodies.push_back(Particle());
    readInitialBodyValuesFromFile(bodies, argv[1]);
    particleCount = bodies.size();


    double start_time, stop_time, time, totalTimeComputeForces, totalTimeMoveBodies, totalGlobalTime = 0.0f;

    //cannot set parallelization in this loop since each loop depends
    // definetly from the bodies state of the immediate loop before.
    for (int iteration = 0; iteration < maxIteration; ++iteration)
    {
        start_time = omp_get_wtime();
        ComputeForces(bodies, gTerm, deltaT);
        stop_time = omp_get_wtime();
        time = stop_time - start_time;
        totalTimeComputeForces += time;

        cout << "\ntime computeForces \t\t: " << iteration + 1 << "\t-\t" << time;

        start_time = omp_get_wtime();
        MoveBodies(bodies, deltaT);
        stop_time = omp_get_wtime();
        time = stop_time - start_time;
        totalTimeMoveBodies += time;

        cout << "\ntime moveBodies \t\t: " << iteration + 1 << "\t-\t" << time;


        fileOutput.str(std::string());
        fileOutput << "out/nbody_" << iteration << ".txt";
        PersistPositions(fileOutput.str(), bodies);
    }

    //outputTotal time taken for computation of all loops
    cout << "\ntotal time computing forces: \t\t" << totalTimeComputeForces <<  " seconds";
    cout << "\ntotal time move bodies: \t\t" << totalTimeMoveBodies <<  " seconds";

    totalGlobalTime = totalTimeComputeForces + totalTimeMoveBodies;

    printf("Total exec time: %f seconds \t\t Average exec time %f seconds\n",  totalGlobalTime, totalGlobalTime/maxIteration);



	return 0;
}