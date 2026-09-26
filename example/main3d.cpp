/*
 * Beautiful 3D manifold example with a "forming" animation (OpenFPM)
 *
 * This example demonstrates how to use the distributed vector `vector_dist`
 * in 3D to sample points on a beautiful surface embedded in 3D space: a
 * TORUS (the doughnut shape). A torus is the canonical "beautiful manifold"
 * -- a 2-manifold embedded in R^3.
 *
 * On top of that, this version writes a TIME SERIES of VTK files so that
 * ParaView can play an animation in which the torus is formed little by little:
 * each frame reveals a larger arc of the main ring (the angle u grows from 0
 * to 2*PI), until the full donut appears.
 *
 * What this example does:
 *   - Initialize the OpenFPM library (MPI, etc.)
 *   - Define a 3D domain (Box), non-periodic boundary conditions, no Ghost
 *   - Pre-generate the (u,v) parameters of all particles ONCE and sort them by
 *     u, so that every frame simply reveals a prefix of the list: the arc grows
 *     smoothly and particles do not jump around between frames
 *   - For each animation frame k = 0 .. NUM_STEPS-1:
 *       * create a distributed vector of particles covering the arc u in [0, u_k]
 *       * place each particle on the torus surface
 *       * color it with a rainbow gradient around the ring (u)
 *       * store the surface normal as a vector property
 *       * write one VTK time step with `write_frame`, which embeds the time
 *         value in the .pvtp: torus_<k>.pvtp + torus_<rank>_<k>.vtp
 *   - Write a ParaView collection file `torus.pvd` that references every
 *     time step and its time value
 *   - Finalize the library
 *
 * In ParaView: open `torus.pvd` (NOT a single .pvtp) and press the Play button
 * (or use the Animation view) to watch the torus form. Color by "color" and add
 * Glyphs using the "normal" vector to see the surface and its orientation.
 *
 * Build & run: see the Makefile (`make main3d` then `make run3d`).
 */

#include <stddef.h>
#include "Vector/vector_dist.hpp"
#include <cmath>
#include <string>
#include <cstdlib>
#include <vector>
#include <utility>
#include <algorithm>
#include <fstream>

int main(int argc, char* argv[])
{
	// Initialize the OpenFPM library (sets up MPI, etc.)
	openfpm_init(&argc, &argv);

	// Define a 3D domain as a box with bounds [0,0,0] - [1,1,1]
	Box<3, float> domain({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});

	// Boundary conditions: non-periodic in x, y and z (the torus is a closed
	// surface that fits entirely inside the domain, so no wrapping is needed).
	size_t bc[3] = {NON_PERIODIC, NON_PERIODIC, NON_PERIODIC};

	// Ghost layer: we do not perform neighbor searches, so no ghost is needed.
	Ghost<3, float> g(0.0);

	// Total number of particles on the fully formed torus.
	const size_t N = 20000;

	// Number of animation frames. The torus is revealed progressively across
	// these frames (one VTK file set per frame).
	const int NUM_STEPS = 60;

	// Torus geometry (kept safely inside the [0,1]^3 domain, centered at 0.5).
	const float PI = 3.14159265358979323846f;
	const float R  = 0.32f;  // major radius (center of tube to torus center)
	const float r  = 0.13f;  // minor radius (radius of the tube)

	// Indices into the aggregate for the two properties
	const int color  = 0;
	const int normal = 1;

	// Vcluster gives us the MPI rank and the total number of ranks.
	auto & v_cl = create_vcluster();

	// Pre-generate the (u, v) parameters of all N particles ONCE, then sort them
	// by u. Because every frame only reveals a prefix of this fixed list, the
	// revealed arc grows smoothly and the particles already shown do not jump
	// around between frames (they keep the same position and color).
	std::vector<std::pair<float, float>> params(N);
	for (size_t i = 0; i < N; ++i)
	{
		float u = 2.0f * PI * ((float)rand() / RAND_MAX);
		float v = 2.0f * PI * ((float)rand() / RAND_MAX);
		params[i] = std::make_pair(u, v);
	}
	std::sort(params.begin(), params.end(),
	          [](const std::pair<float, float> & a, const std::pair<float, float> & b)
	          { return a.first < b.first; });

	// (file name, time value) of every written time step, used to build the
	// ParaView collection file `torus.pvd` after the loop.
	std::vector<std::pair<std::string, double>> pvd_steps;

	// Produce one frame at a time. Each frame reveals a larger arc of the ring
	// angle u, so playing the frames back forms the torus little by little.
	for (int k = 0; k < NUM_STEPS; ++k)
	{
		// Fraction of the ring revealed in this frame (0 .. 1).
		float fraction = (float)(k + 1) / (float)NUM_STEPS;

		// Number of particles present in this (partial) frame. This is the size
		// of the global prefix of the sorted (u, v) list that we reveal.
		size_t total_count = (size_t)(N * fraction);
		if (total_count < 1) total_count = 1;

		// `vector_dist(total_count, ...)` splits `total_count` across ranks
		// itself: rank r receives base + 1 particles when r < remainder, and
		// base otherwise. Mirror that same partition here so each rank knows
		// which slice of the globally sorted `params` array it owns.
		size_t np   = v_cl.getProcessingUnits();
		size_t rank = v_cl.getProcessUnitID();
		size_t base = total_count / np;
		size_t rem  = total_count % np;
		size_t offset = base * rank + (rank < rem ? rank : rem);

		// Create a distributed vector with:
		//   - 3           : dimensionality of space
		//   - float       : type used for spatial coordinates
		//   - aggregate<...> : properties stored per particle
		//       * float       : a scalar "color" property (rainbow around ring)
		//       * float[3]    : the surface normal (a vector property)
		vector_dist<3, float, aggregate<float, float[3]>> vd(total_count, domain, bc, g);

		// Parameters used to write a ParaView collection file (rank 0 only).
		double time_value = (double)k;
		openfpm::vector<std::string> names({"color", "normal"});
		vd.setPropNames(names);

		// Assign a position on the (partial) torus surface to each particle.
		size_t local_idx = 0;
		auto it = vd.getDomainIterator();
		while (it.isNext())
		{
			auto p = it.get();

			// The particles owned by this rank are the slice
			// [offset, offset + p_np) of the globally sorted (u, v) list.
			float u = params[offset + local_idx].first;
			float v = params[offset + local_idx].second;
			++local_idx;

			// Parametric torus surface, then shifted to the center of the domain.
			float x = (R + r * cos(v)) * cos(u);
			float y = (R + r * cos(v)) * sin(u);
			float z = r * sin(v);

			vd.getPos(p)[0] = x + 0.5f;
			vd.getPos(p)[1] = y + 0.5f;
			vd.getPos(p)[2] = z + 0.5f;

			// Rainbow color around the main ring (u goes 0..1 around the torus).
			vd.template getProp<color>(p) = u / (2.0f * PI);

			// Outward surface normal of the torus at this point.
			float nx = cos(v) * cos(u);
			float ny = cos(v) * sin(u);
			float nz = sin(v);
			vd.template getProp<normal>(p)[0] = nx;
			vd.template getProp<normal>(p)[1] = ny;
			vd.template getProp<normal>(p)[2] = nz;

			++it;
		}

		// Redistribute particles according to the underlying domain decomposition.
		vd.map();

		// Write this time step. `write_frame(out, iteration, time)` produces
		//   - torus_<rank>_<iteration>.vtp   (per-rank geometry)
		//   - torus_<iteration>.pvtp         (collection with TimeValue metadata)
		// The TimeValue is what lets ParaView build a real time axis.
		vd.write_frame("torus", k, time_value);

		// Rank 0 records this step in the ParaView collection file description;
		// the file itself is written once, after the loop.
		if (v_cl.getProcessUnitID() == 0)
		{
			pvd_steps.push_back(std::make_pair(
				"torus_" + std::to_string(k) + ".pvtp", time_value));
		}
	}

	// Write the ParaView collection file `torus.pvd`. Opening THIS file in
	// ParaView gives a proper time series (correct time axis, correct frame
	// count), so the Play button animates the torus forming instead of running
	// "really fast" or appearing to do nothing.
	if (v_cl.getProcessUnitID() == 0)
	{
		std::ofstream pvd("torus.pvd");
		pvd << "<?xml version=\"1.0\"?>\n";
		pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
		pvd << "  <Collection>\n";
		for (size_t i = 0; i < pvd_steps.size(); ++i)
		{
			pvd << "    <DataSet timestep=\"" << pvd_steps[i].second
			    << "\" group=\"\" part=\"0\" file=\"" << pvd_steps[i].first << "\"/>\n";
		}
		pvd << "  </Collection>\n";
		pvd << "</VTKFile>\n";
	}

	// Finalize the library
	openfpm_finalize();

	return 0;
}
