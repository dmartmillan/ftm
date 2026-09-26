/*
 * Simple OpenFPM example
 *
 * This example demonstrates the basic usage of the distributed vector
 * `vector_dist` from OpenFPM:
 *   - Initialize the library
 *   - Define a domain (Box), boundary conditions, and a Ghost layer
 *   - Create a distributed vector of particles
 *   - Assign random positions and properties to the particles
 *   - Redistribute particles across processors (map)
 *   - Reduce a counter across all processors (Vcluster)
 *   - Write the result to VTK files (visualizable with ParaView)
 *   - Finalize the library
 *
 * Build & run instructions are in the README at the workspace root.
 */

#include <stddef.h>
#include "Vector/vector_dist.hpp"

int main(int argc, char* argv[])
{
	// Initialize the OpenFPM library (sets up MPI, etc.)
	openfpm_init(&argc, &argv);

	// Define a 2D domain as a box with bounds [0,0] - [1,1]
	Box<2, float> domain({0.0, 0.0}, {1.0, 1.0});

	// Boundary conditions: periodic in both x and y
	size_t bc[2] = {PERIODIC, PERIODIC};

	// Ghost layer: extension of the ghost part in physical units
	Ghost<2, float> g(0.01);

	// Create a distributed vector with:
	//   - 2           : dimensionality of space
	//   - float       : type used for spatial coordinates
	//   - aggregate<...> : properties stored per particle
	//       * float       : a scalar property
	//       * float[3]    : a vector property
	//       * float[3][3] : a tensor (rank-2) property
	// 4096 particles are created (split across processors).
	vector_dist<2, float, aggregate<float, float[3], float[3][3]>> vd(4096, domain, bc, g);

	// Indices into the aggregate for the three properties
	const int scalar = 0;
	const int vector = 1;
	const int tensor = 2;

	// Assign a random position to each particle
	auto it = vd.getDomainIterator();
	while (it.isNext())
	{
		auto key = it.get();

		// Random position in [0,1] for x and y
		vd.getPos(key)[0] = (float)rand() / RAND_MAX;
		vd.getPos(key)[1] = (float)rand() / RAND_MAX;

		++it;
	}

	// Redistribute particles according to the underlying domain decomposition
	vd.map();

	// Assign values to the particle properties
	size_t cnt = 0;
	it = vd.getDomainIterator();
	while (it.isNext())
	{
		auto p = it.get();

		// scalar property
		vd.template getProp<scalar>(p) = 1.0;

		// vector property
		vd.template getProp<vector>(p)[0] = 1.0;
		vd.template getProp<vector>(p)[1] = 1.0;
		vd.template getProp<vector>(p)[2] = 1.0;

		// tensor property
		vd.template getProp<tensor>(p)[0][0] = 1.0;
		vd.template getProp<tensor>(p)[0][1] = 1.0;
		vd.template getProp<tensor>(p)[0][2] = 1.0;
		vd.template getProp<tensor>(p)[1][0] = 1.0;
		vd.template getProp<tensor>(p)[1][1] = 1.0;
		vd.template getProp<tensor>(p)[1][2] = 1.0;
		vd.template getProp<tensor>(p)[2][0] = 1.0;
		vd.template getProp<tensor>(p)[2][1] = 1.0;
		vd.template getProp<tensor>(p)[2][2] = 1.0;

		cnt++;
		++it;
	}

	// Reduce (sum) the local counter across all processors to get the total
	auto & v_cl = create_vcluster();
	v_cl.sum(cnt);
	v_cl.execute();

	if (v_cl.getProcessUnitID() == 0)
	{
		std::cout << "Total number of particles across all processors: " << cnt << std::endl;
	}

	// Write the particles to VTK files (one per processor) for visualization
	openfpm::vector<std::string> names({"scalar", "vector", "tensor"});
	vd.setPropNames(names);
	vd.write("particles");

	// Finalize the library
	openfpm_finalize();

	return 0;
}
