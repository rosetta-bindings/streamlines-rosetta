#!/usr/bin/env python3
"""Trace the streamlines of a vector property carried by a Gocad TSurf and show
them in 3D.

    python scripts/plot.py extern/streamlines/data/Faults.ts --property U

Run with `--list` to see the properties a file carries and their sizes.
"""

import argparse
import os
import sys

import numpy as np

# The binding is not necessarily pip-installed; fall back to the in-tree module
# dropped next to its sources by the CMake build.
try:
    import streamlines
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(_here), "bindings", "nanobind"))
    import streamlines


# ---------------------------------------------------------------------------
# Gocad TSurf reader
# ---------------------------------------------------------------------------

def load_gocad_ts(filename):
    """Read a Gocad TSurf ASCII file.

    Returns a list of parts, one per TFACE, each a dict:
        {"name": str,
         "vertices":   (nv, 3) float array,
         "triangles":  (nt, 3) int array,
         "properties": {name: (nv, esize) float array}}

    Triangle indices are renumbered 0..nv-1 *within each part* (Gocad vertex ids
    are file-global and may start at 0 or 1).

    Supported records: GOCAD TSurf, HEADER { name: ... }, PROPERTIES, ESIZES,
    NO_DATA_VALUES, TFACE, VRTX/PVRTX, ATOM/PATOM (vertex aliasing) and TRGL.
    Everything else (PROPERTY_CLASSES, borders, BSTONE, ...) is ignored.
    """
    default_name = os.path.splitext(os.path.basename(filename))[0]
    # vertex id -> (x, y, z) and vertex id -> [property values]. Ids are shared
    # by the TFACEs of one TSurf object but restart from scratch at the next
    # `GOCAD` line, so these are per object.
    coords = {}
    props = {}
    parts = []        # one entry per TFACE
    part = None
    obj_name = default_name  # HEADER of the TSurf currently being read
    prop_names = []          # PROPERTIES of the TSurf currently being read
    prop_sizes = []          # matching ESIZES (1 per property when absent)
    no_data = []
    in_header = False

    def new_part():
        # `order` keeps the vertices in declaration order so the part survives a
        # write/read round-trip unpermuted
        parts.append({"name": obj_name, "coords": coords, "props": props,
                      "prop_names": prop_names, "prop_sizes": prop_sizes,
                      "no_data": no_data, "order": [], "tris": []})
        return parts[-1]

    with open(filename) as f:
        for lineno, line in enumerate(f, 1):
            tok = line.split()
            if not tok:
                continue
            key = tok[0].upper()

            if in_header:
                if key == "}":
                    in_header = False
                elif ":" in line:
                    k, _, v = line.partition(":")
                    v = v.strip().lstrip("=").strip()
                    if k.strip().lower() == "name" and v:
                        # the HEADER precedes the TFACEs of its object, so it
                        # names the parts still to come, not the previous one
                        obj_name = v
                continue

            if key == "GOCAD":
                obj_name = default_name
                coords, props = {}, {}
                prop_names, prop_sizes, no_data = [], [], []
                part = None
                continue
            if key == "HEADER":
                in_header = "}" not in line
                continue
            if key == "PROPERTIES":
                prop_names = tok[1:]
                # ESIZES may come before or after PROPERTIES; default to scalars
                if len(prop_sizes) != len(prop_names):
                    prop_sizes = [1] * len(prop_names)
                continue
            if key == "ESIZES":
                prop_sizes = [int(t) for t in tok[1:]]
                continue
            if key == "NO_DATA_VALUES":
                no_data = [float(t) for t in tok[1:]]
                continue
            if key == "TFACE":
                part = new_part()
                continue
            if key == "END":
                part = None
                continue

            if key in ("VRTX", "PVRTX", "ATOM", "PATOM"):
                gid = int(tok[1])
                if key in ("VRTX", "PVRTX"):
                    coords[gid] = (float(tok[2]), float(tok[3]), float(tok[4]))
                    values = tok[5:]
                else:
                    # an ATOM is a vertex sharing the position of another one; it
                    # may either redefine the properties or inherit them
                    ref = int(tok[2])
                    coords[gid] = coords[ref]
                    values = tok[3:]
                    if not values and ref in props:
                        props[gid] = props[ref]
                if values:
                    props[gid] = [float(v) for v in values]
                if part is None:  # geometry before any TFACE
                    part = new_part()
                part["order"].append(gid)
            elif key == "TRGL":
                if len(tok) < 4:
                    raise ValueError(f"{filename}:{lineno}: malformed TRGL")
                if part is None:
                    part = new_part()
                part["tris"].append(tuple(int(t) for t in tok[1:4]))

    result = []
    for p in parts:
        if not p["tris"]:
            continue
        # Gocad keeps unused vertices around, and a TRGL may reference a vertex
        # declared under another TFACE — so index the referenced ones only,
        # declaration order first, then any foreign vertex as it appears.
        pcoords = p["coords"]
        used = {g for tri in p["tris"] for g in tri}
        local = {}
        for gid in p["order"]:
            if gid in used and gid not in local:
                local[gid] = len(local)
        for tri in p["tris"]:
            for gid in tri:
                if gid not in local:
                    if gid not in pcoords:
                        raise ValueError(
                            f"{filename}: triangle uses unknown vertex {gid}"
                        )
                    local[gid] = len(local)

        nv = len(local)
        vertices = np.empty((nv, 3))
        for gid, l in local.items():
            vertices[l] = pcoords[gid]
        triangles = np.array([[local[g] for g in tri] for tri in p["tris"]],
                             dtype=np.int32)

        # de-interleave the flat per-vertex value list into one array per property
        names, sizes = p["prop_names"], p["prop_sizes"]
        ncomp = sum(sizes)
        properties = {}
        if names and ncomp:
            flat = np.full((nv, ncomp), np.nan)
            for gid, l in local.items():
                values = p["props"].get(gid)
                if values is not None and len(values) >= ncomp:
                    flat[l] = values[:ncomp]
            offset = 0
            for name, size in zip(names, sizes):
                block = flat[:, offset:offset + size]
                offset += size
                if np.isnan(block).all():
                    continue  # property declared but never valued on this part
                properties[name] = block
            for name, ndv in zip(names, p["no_data"]):
                if name in properties:
                    properties[name][properties[name] == ndv] = np.nan

        result.append({"name": p["name"], "vertices": vertices,
                       "triangles": triangles, "properties": properties})

    if not result:
        raise ValueError(f"{filename}: no triangulated surface found")
    return result


def select_vectors(part, spec):
    """Resolve `spec` to an (nv, 3) vector field of `part`.

    `spec` is either the name of a size-3 property, or three comma-separated
    names of scalar properties to assemble ("Dx,Dy,Dz").
    """
    properties = part["properties"]
    names = [s.strip() for s in spec.split(",")]

    if len(names) == 3:
        missing = [n for n in names if n not in properties]
        if missing:
            raise KeyError(f"unknown property {', '.join(missing)}")
        bad = [n for n in names if properties[n].shape[1] != 1]
        if bad:
            raise KeyError(f"{', '.join(bad)}: expected scalar components")
        return np.hstack([properties[n] for n in names])

    if len(names) != 1:
        raise KeyError(f"{spec}: give one vector property or three scalar ones")

    name = names[0]
    if name not in properties:
        raise KeyError(f"unknown property {name}")
    field = properties[name]
    if field.shape[1] != 3:
        raise KeyError(f"{name}: has {field.shape[1]} component(s), not 3")
    return field


def describe(parts):
    """One line per part listing its size, then one per property."""
    out = []
    for i, p in enumerate(parts):
        out.append(f"[{i}] {p['name']}: {len(p['vertices'])} vertices, "
                   f"{len(p['triangles'])} triangles")
        if not p["properties"]:
            out.append("      (no properties)")
        for name, field in p["properties"].items():
            kind = "vector3" if field.shape[1] == 3 else f"{field.shape[1]} comp."
            out.append(f"      {name:<24} {kind}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Streamline generation
# ---------------------------------------------------------------------------

def make_seeding(args):
    """Build the SeedingStrategy asked for, or None for the library default."""
    if args.seeding == "all":
        return None
    if args.seeding == "density":
        return streamlines.DensitySeeding(args.seeding_param)
    if args.seeding == "threshold":
        return streamlines.ThresholdSeeding(args.seeding_param)
    if args.seeding == "probability":
        return streamlines.ProbabilitySeeding(args.seeding_param)
    raise ValueError(f"unknown seeding strategy {args.seeding}")


def generate_streamlines(vertices, triangles, vectors, scalars, args):
    """Trace the streamlines of `vectors` over the surface.

    Returns a list of (n, 3) arrays, one per streamline.
    """
    mesh = streamlines.TriMesh()
    mesh.set_vertices(vertices.ravel().astype(np.float64).tolist())
    mesh.set_triangles(triangles.ravel().astype(np.int32).tolist())
    mesh.set_vectors(vectors.ravel().astype(np.float64).tolist())
    if scalars is not None:
        mesh.set_scalars(scalars.astype(np.float64).tolist())

    if not args.no_project:
        # the field read from the file has no reason to be tangent to the
        # surface, and the generator integrates inside the triangle planes
        mesh.project_vectors_to_surface()
    mesh.build_adjacency()

    params = streamlines.StreamlineParams()
    params.density = args.density
    params.max_iterations = args.max_iterations
    params.random_seed = args.random_seed
    if args.separation is not None:
        params.separation_distance = args.separation
    if args.step is not None:
        params.integration_step = args.step
    strategy = make_seeding(args)
    if strategy is not None:
        params.seeding_strategy = strategy

    generator = streamlines.StreamlineGenerator3D(mesh, params)
    lines = []
    for line in generator.generate():
        points = np.asarray(line.get_points_flat()).reshape(-1, 3)
        if len(points) > 1:
            lines.append(points)
    return lines


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def lines_to_polydata(lines):
    """Pack a list of (n, 3) polylines into a single pyvista.PolyData."""
    import pyvista as pv

    cells, offset = [], 0
    for line in lines:
        n = len(line)
        cells.append(np.concatenate(([n], np.arange(offset, offset + n))))
        offset += n
    return pv.PolyData(np.vstack(lines), lines=np.concatenate(cells))


def plot(parts, args):
    import pyvista as pv

    plotter = pv.Plotter(off_screen=args.save is not None)
    total = 0

    for part in parts:
        vertices, triangles = part["vertices"], part["triangles"]
        # a vertex with no value would poison the interpolation; the mesh is the
        # file's, so zero the field there rather than drop the geometry
        vectors = np.nan_to_num(select_vectors(part, args.property))
        magnitude = np.linalg.norm(vectors, axis=1)

        scalars = None
        if args.seed_scalar == "magnitude":
            scalars = magnitude
        elif args.seed_scalar:
            field = part["properties"].get(args.seed_scalar)
            if field is None:
                raise KeyError(f"unknown property {args.seed_scalar}")
            scalars = np.nan_to_num(field[:, 0])
        elif args.seeding != "all":
            # every strategy but AllTrianglesSeeding reads the scalar field
            scalars = magnitude

        lines = generate_streamlines(vertices, triangles, vectors, scalars, args)
        total += len(lines)
        print(f"{part['name']}: {len(lines)} streamlines")

        faces = np.hstack([np.full((len(triangles), 1), 3), triangles]).ravel()
        surface = pv.PolyData(vertices, faces)
        # smooth_shading interpolates the vertex normals across each triangle, so
        # the coarse fault meshes stop reading as facets
        plotter.add_mesh(surface, color=args.surface_color, opacity=args.opacity,
                         show_edges=args.show_edges, edge_color="grey",
                         smooth_shading=True)

        if not lines:
            continue
        polylines = lines_to_polydata(lines)
        if args.tube:
            polylines = polylines.tube(radius=args.tube)
        plotter.add_mesh(polylines, color=args.line_color,
                         line_width=args.line_width)

    print(f"total: {total} streamlines")
    plotter.add_axes()
    if args.bounds:
        plotter.show_bounds(grid="back", location="outer")
    if args.save:
        plotter.show(screenshot=args.save)
        print(f"wrote {args.save}")
    else:
        plotter.show()


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="Gocad TSurf (.ts) file")
    p.add_argument("-p", "--property",
                   help="vector3 property to trace, or three scalar ones "
                        "(\"Dx,Dy,Dz\"). Defaults to the only vector3 property.")
    p.add_argument("-l", "--list", action="store_true",
                   help="list the parts and properties of the file and exit")
    p.add_argument("--part", type=int, action="append",
                   help="index of a part to plot (repeatable; default: all)")

    g = p.add_argument_group("streamlines")
    g.add_argument("--density", type=float, default=1.0,
                   help="spacing of the streamlines: >1 denser, <1 sparser (1.0)")
    g.add_argument("--separation", type=float,
                   help="separation distance in model units (overrides --density)")
    g.add_argument("--step", type=float,
                   help="integration step (default: derived from the bounding box)")
    g.add_argument("--max-iterations", type=int, default=1000,
                   help="max integration steps per direction (1000)")
    g.add_argument("--seeding", default="all",
                   choices=["all", "density", "threshold", "probability"],
                   help="seeding strategy (all)")
    g.add_argument("--seeding-param", type=float, default=1.0,
                   help="density factor / threshold / exponent of --seeding (1.0)")
    g.add_argument("--seed-scalar",
                   help="scalar property driving the seeding, or \"magnitude\" "
                        "for |vector| (the default once --seeding is set)")
    g.add_argument("--random-seed", type=int, default=0,
                   help="RNG seed of the probability seeding, 0 for random (0)")
    g.add_argument("--no-project", action="store_true",
                   help="skip projecting the vectors onto the tangent planes")

    g = p.add_argument_group("display")
    g.add_argument("--surface-color", default="lightgrey",
                   help="surface colour (lightgrey)")
    g.add_argument("--opacity", type=float, default=1.0, help="surface opacity (1.0)")
    g.add_argument("--show-edges", action="store_true", help="draw the mesh edges")
    g.add_argument("--line-color", default="black", help="streamline colour (black)")
    g.add_argument("--line-width", type=float, default=2.0,
                   help="streamline width (2)")
    g.add_argument("--tube", type=float,
                   help="draw the streamlines as tubes of this radius")
    g.add_argument("--bounds", action="store_true",
                   help="draw the labelled bounding box")
    g.add_argument("--save",
                   help="render off-screen to this image instead of showing")

    args = p.parse_args(argv)

    parts = load_gocad_ts(args.file)
    if args.list:
        print(describe(parts))
        return 0

    if args.part:
        try:
            parts = [parts[i] for i in args.part]
        except IndexError:
            p.error(f"--part out of range, the file has {len(parts)} part(s)")

    if not args.property:
        candidates = {name for part in parts
                      for name, field in part["properties"].items()
                      if field.shape[1] == 3}
        if len(candidates) != 1:
            p.error(
                "--property is required: "
                + (f"found {len(candidates)} vector3 properties "
                   f"({', '.join(sorted(candidates))})" if candidates
                   else "no vector3 property in this file")
                + "\n\n" + describe(parts))
        args.property = candidates.pop()

    # a part without the property is not an error — a TSurf file often holds
    # several objects and only some of them carry the field
    kept = []
    for part in parts:
        try:
            select_vectors(part, args.property)
        except KeyError as e:
            print(f"skipping {part['name']}: {e}", file=sys.stderr)
            continue
        kept.append(part)
    if not kept:
        p.error(f"no part carries {args.property}\n\n{describe(parts)}")

    plot(kept, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
