//----------------------------------------------------------------------------------------------
//	Filename:	nbody.cpp
//	Author:		Keith Bugeja
//----------------------------------------------------------------------------------------------
//  CPS3227 assignment for academic year 2017/2018:
//	Sample naive [O(n^2)] implementation for the N-Body problem.
//----------------------------------------------------------------------------------------------

#include <mpi.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <string.h>
#include <stdlib.h>


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

const int maxIteration = 1000;
const float deltaT = 0.01f;
const float gTerm = 20.f;


int my_rank;
int size;
int* partition_start;
int* partition_end;

#define	NUM_THREADS	12

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

struct FlatParticle{
    float px, py;
    float vx, vy;
    float mass;

    FlatParticle() {}

    FlatParticle(float p_x, float p_y, float v_x, float v_y, float m)
            : px(p_x), py(p_y), vx(v_x), vy(v_y),mass(m) {}

    FlatParticle(Particle p_particle)
            : px(p_particle.Position.X), py(p_particle.Position.Y), vx(p_particle.Velocity.X), vy(p_particle.Velocity.Y),mass(p_particle.Mass) {}


};


MPI_Datatype MPI_FLAT_PARTICLE;
MPI_Status status;


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
    for (j = partition_start[my_rank]; j < partition_end[my_rank]; j++)
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
void MoveBodies(std::vector<Particle> &p_bodies,  float p_deltaT)
{
    omp_set_num_threads(NUM_THREADS);
    size_t j;
    int p_bodies_size = p_bodies.size();
    //static scheduling was done because of know number
    #if DO_PARALLELIZE == 1
        #pragma omp parallel for shared(p_bodies,p_deltaT,p_bodies_size) private(j) schedule(static) num_threads(NUM_THREADS)
    #endif
    for (j = partition_start[my_rank]; j < partition_end[my_rank]; j++)
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

/*
 * Used to copy particle vector to flat particle array.
 */
FlatParticle* CopyParticleVectorToFlatParticleArray( std::vector<Particle> &p_bodies)
{
    FlatParticle* flatParticles = (FlatParticle *) malloc(sizeof(FlatParticle) * p_bodies.size());
    memset(flatParticles, 0, sizeof(FlatParticle) * p_bodies.size());

    //intialize bodies
    for (int bodyIndex = 0; bodyIndex < p_bodies.size(); bodyIndex++)
    {
        //Particle particle = Particle();
        FlatParticle flatParticle(p_bodies[bodyIndex]);

        //p_bodies.push_back(particle);
        flatParticles[bodyIndex] = flatParticle;
    }

    return flatParticles;
}

/*
 * Used to copy flat particle collection to Particle vector
 */
void CopyFlatParticleArrayToParticleVector( std::vector<Particle> &p_bodies, FlatParticle* &flatParticles)
{
    //intialize bodies
    for (int bodyIndex = 0; bodyIndex < p_bodies.size(); bodyIndex++)
    {
        Particle particle;
        particle.Position.X = flatParticles[bodyIndex].px;
        particle.Position.Y = flatParticles[bodyIndex].py;
        particle.Velocity.X = flatParticles[bodyIndex].vx;
        particle.Velocity.Y = flatParticles[bodyIndex].vy;
        particle.Mass = flatParticles[bodyIndex].mass;

        p_bodies[bodyIndex] = particle;
    }
}


void runMpiParallelization(int processors, std::vector<Particle> &p_bodies)
{
    int i, j;
    int particleCount = p_bodies.size();

    FlatParticle* flatParticles = CopyParticleVectorToFlatParticleArray(p_bodies);


    for (i = 0; i < processors; i++)
    {
        MPI_Bcast(flatParticles, particleCount, MPI_FLAT_PARTICLE, 0, MPI_COMM_WORLD);


        CopyFlatParticleArrayToParticleVector(p_bodies, flatParticles);

        ComputeForces(p_bodies, gTerm, deltaT);
        MoveBodies(p_bodies, deltaT);

        FlatParticle* flatParticles2 = CopyParticleVectorToFlatParticleArray( p_bodies);

        if (my_rank == 0)
        {
            for (j = 1; j < size; j++)
                MPI_Recv(&flatParticles[partition_start[j]],
                         partition_end[j] - partition_start[j],
                         MPI_FLAT_PARTICLE, j, 100, MPI_COMM_WORLD, &status);

        }
        else
        {
            //send back the modified bodies after computation to process 0
            MPI_Send(&flatParticles[partition_start[my_rank]],
                     partition_end[my_rank] - partition_start[my_rank],
                     MPI_FLAT_PARTICLE, 0, 100, MPI_COMM_WORLD);
        }
    }
}


int main(int argc, char **argv)
{
    int i, z;
    int particleCount;
    double startTime, endTime, totalGlobalTime, averageTime;

    double totalTime =0.f;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Type_contiguous(5, MPI_FLOAT, &MPI_FLAT_PARTICLE);
    MPI_Type_commit(&MPI_FLAT_PARTICLE);


    char hn[256];
    gethostname(hn, 256);


    printf("output process %d of %d host %s\n",  my_rank, size, hn);


	std::stringstream fileOutput;
	std::vector<Particle> bodies;


    readInitialBodyValuesFromFile(bodies, argv[1]);
    particleCount = bodies.size();

    //partition_start and partition_end contain information on how to partition the problem
    partition_start = (int *) malloc(size * sizeof(int));
    partition_end   = (int *) malloc(size * sizeof(int));


    partition_start[0] = 0;
    partition_end[0] = particleCount / size;
    for (i = 1; i < size; i++)
    {
        partition_start[i] = partition_end[i - 1];
        partition_end[i] = partition_start[i] + particleCount / size;
    }
    partition_end[size - 1] = particleCount;


    for (int iteration = 0; iteration < maxIteration; ++iteration)
    {
        startTime = MPI_Wtime();
        runMpiParallelization(size, bodies);
        endTime = MPI_Wtime();
        totalTime = endTime - startTime;
        totalGlobalTime += totalTime;
        printf("iteration no.: %d took %f seconds\n",  iteration, totalTime);


        if (my_rank == 0)
        {
            fileOutput.str(std::string());
            fileOutput << "out/nbody_" << iteration << ".txt";
            PersistPositions(fileOutput.str(), bodies);
        }
    }

    averageTime = totalGlobalTime/maxIteration;
    printf("Total exec time: %f seconds \t\t Average exec time %f seconds\n",  totalGlobalTime, averageTime);

    MPI_Finalize();
	return 0;
}