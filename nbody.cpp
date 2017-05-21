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
#define	NUM_THREADS	8

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

    #pragma omp declare reduction(plusVector2 : Vector2 : omp_out = omp_out + omp_in) \
             initializer (omp_priv= 0.f)

    force = 0.f, acceleration = 0.f;

    #pragma omp parallel for schedule(static) private(j,acceleration) shared(p_gravitationalTerm) num_threads(NUM_THREADS)
    for (j = 0; j < p_bodies.size(); ++j)
    {
        Particle &p1 = p_bodies[j];

        #pragma omp parallel for schedule(static) private(k,direction,distance) reduction(plusVector2: force)
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
    #pragma omp parallel for shared(p_bodies,p_deltaT,p_bodies_size) private(j) schedule(static) num_threads(8)
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
	std::cout << "\nWriting to file: " << p_strFilename << std::endl;
	std::ofstream output(p_strFilename.c_str());
	
	if (output.is_open())
	{	
		for (int j = 0; j < p_bodies.size(); j++)
		{
			output << 	p_bodies[j].Mass << ", " <<
				p_bodies[j].Position.Element[0] << ", " <<
				p_bodies[j].Position.Element[1] << std::endl;
		}
		
		output.close();
	}
	else
		std::cerr << "Unable to persist data to file:" << p_strFilename << std::endl;

}

int main(int argc, char **argv)
{
	const int particleCount = 1024; // TODO - to test with 1024
	const int maxIteration = 1000; //TODO - to test with 1000
	const float deltaT = 0.01f;
	const float gTerm = 20.f;
    double totalTime = 0;

	std::stringstream fileOutput;
	std::vector<Particle> bodies;

    //TODO - clean
    int milliSecondsDelay =5000;


	for (int bodyIndex = 0; bodyIndex < particleCount; ++bodyIndex)
		bodies.push_back(Particle());

    //cannot set parallelization in this loop since each loop depends
    // definetly from the bodies state of the immediate loop before.
    for (int iteration = 0; iteration < maxIteration; ++iteration)
    {

        ComputeForces(bodies, gTerm, deltaT);

        //TODO - should consider to put a barrier here.
        double start_time = omp_get_wtime();
        MoveBodies(bodies, deltaT);
        double stop_time = omp_get_wtime();
        double time = stop_time - start_time;
        totalTime += time;

        cout << "\ncheck iteration time " << iteration + 1 << ":\t\t" << time;


        fileOutput.str(std::string());
        fileOutput << "out/nbody_" << iteration << ".txt";

        PersistPositions(fileOutput.str(), bodies);
    }

    //outputTotal time taken for computation of all loops
    cout << "\ntotal time: \t\t" << totalTime <<  " seconds"  << endl;


	return 0;
}