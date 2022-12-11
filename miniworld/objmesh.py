import os

import numpy as np
import pyglet
from pyglet.gl import GL_TEXTURE_2D, GL_TRIANGLES, glBindTexture, glDisable, glEnable

from miniworld.opengl import Texture
from miniworld.utils import get_file_path


class ObjMesh:
    """
    Load and render OBJ model files
    """

    # Loaded mesh files, indexed by mesh file path
    cache = {}

    @classmethod
    def get(self, mesh_name):
        """
        Load a mesh or used a cached version
        """

        # Assemble the absolute path to the mesh file
        file_path = get_file_path("meshes", mesh_name, "obj")

        if file_path in self.cache:
            return self.cache[file_path]

        mesh = ObjMesh(file_path)
        self.cache[file_path] = mesh

        return mesh

    def __init__(self, file_path):
        """
        Load an OBJ model file

        Limitations:
        - only one object/group
        - only triangle faces
        """

        # OBJ file format:
        # #Comments
        # mtllib file_name
        # o object_name
        # v x y z
        # vt u v
        # vn x y z
        # usemtl mtl_name
        # f v0/t0/n0 v1/t1/n1 v2/t2/n2

        # print('Loading mesh "%s"' % file_path)

        # Attempt to load the materials library
        materials = self._load_mtl(file_path)
        mesh_file = open(file_path)

        verts = []
        texs = []
        normals = []
        faces = []

        cur_mtl = ""

        # For each line of the input file
        for line in mesh_file:
            line = line.rstrip(" \r\n")

            # Skip comments
            if line.startswith("#") or line == "":
                continue

            tokens = line.split(" ")
            tokens = map(lambda t: t.strip(" "), tokens)
            tokens = list(filter(lambda t: t != "", tokens))

            prefix = tokens[0]
            tokens = tokens[1:]

            if prefix == "v":
                vert = list(map(lambda v: float(v), tokens))
                verts.append(vert)

            if prefix == "vt":
                tc = list(map(lambda v: float(v), tokens))
                texs.append(tc)

            if prefix == "vn":
                normal = list(map(lambda v: float(v), tokens))
                normals.append(normal)

            if prefix == "usemtl":
                mtl_name = tokens[0]
                if mtl_name in materials:
                    cur_mtl = mtl_name
                else:
                    cur_mtl = ""

            if prefix == "f":
                assert len(tokens) == 3, "only triangle faces are supported"

                face = []
                for token in tokens:
                    indices = filter(lambda t: t != "", token.split("/"))
                    indices = list(map(lambda idx: int(idx), indices))
                    assert len(indices) == 2 or len(indices) == 3
                    face.append(indices)

                faces.append([face, cur_mtl])

        mesh_file.close()

        # Sort the faces by material name
        faces.sort(key=lambda f: f[1])

        # Compute the start and end faces for each chunk in the model
        cur_mtl = None
        chunks = []
        for idx, face in enumerate(faces):
            face, mtl_name = face
            if mtl_name != cur_mtl:
                if len(chunks) > 0:
                    chunks[-1]["end_idx"] = idx
                chunks.append(
                    {"mtl": materials[mtl_name], "start_idx": idx, "end_idx": None}
                )
                cur_mtl = mtl_name
        chunks[-1]["end_idx"] = len(faces)

        num_faces = len(faces)
        # print('num verts=%d' % len(verts))
        # print('num faces=%d' % num_faces)
        # print('num chunks=%d' % len(chunks))

        # Create numpy arrays to store the vertex data
        list_verts = np.zeros(shape=(num_faces, 3, 3), dtype=np.float32)
        list_norms = np.zeros(shape=(num_faces, 3, 3), dtype=np.float32)
        list_texcs = np.zeros(shape=(num_faces, 3, 2), dtype=np.float32)
        list_color = np.zeros(shape=(num_faces, 3, 3), dtype=np.float32)

        # For each triangle
        for f_idx, face in enumerate(faces):
            face, mtl_name = face

            # Get the color for this face
            f_mtl = materials[mtl_name]
            f_color = f_mtl["Kd"] if f_mtl else np.array((1, 1, 1))

            # For each tuple of indices
            for l_idx, indices in enumerate(face):
                # Note: OBJ uses 1-based indexing
                # and texture coordinates are optional
                if len(indices) == 3:
                    v_idx, t_idx, n_idx = indices
                    vert = verts[v_idx - 1]
                    texc = texs[t_idx - 1]
                    normal = normals[n_idx - 1]
                else:
                    v_idx, n_idx = indices
                    vert = verts[v_idx - 1]
                    normal = normals[n_idx - 1]
                    texc = [0, 0]

                list_verts[f_idx, l_idx, :] = vert
                list_texcs[f_idx, l_idx, :] = texc
                list_norms[f_idx, l_idx, :] = normal
                list_color[f_idx, l_idx, :] = f_color

        # Re-center the object so that the base is at y=0
        # and the object is centered in x and z
        min_coords = list_verts.min(axis=0).min(axis=0)
        max_coords = list_verts.max(axis=0).min(axis=0)
        mean_coords = (min_coords + max_coords) / 2
        min_y = min_coords[1]
        mean_x = mean_coords[0]
        mean_z = mean_coords[2]
        list_verts[:, :, 1] -= min_y
        list_verts[:, :, 0] -= mean_x
        list_verts[:, :, 2] -= mean_z

        # Recompute the object extents after centering
        self.min_coords = list_verts.min(axis=0).min(axis=0)
        self.max_coords = list_verts.max(axis=0).max(axis=0)

        # Vertex lists, one per chunk
        self.vlists = []

        # Textures, one per chunk
        self.textures = []

        # For each chunk
        for chunk in chunks:
            start_idx = chunk["start_idx"]
            end_idx = chunk["end_idx"]
            num_faces_chunk = end_idx - start_idx

            # Create a vertex list to be used for rendering
            vlist = pyglet.graphics.vertex_list(
                3 * num_faces_chunk,
                ("v3f", list_verts[start_idx:end_idx, :, :].reshape(-1)),
                ("t2f", list_texcs[start_idx:end_idx, :, :].reshape(-1)),
                ("n3f", list_norms[start_idx:end_idx, :, :].reshape(-1)),
                ("c3f", list_color[start_idx:end_idx, :, :].reshape(-1)),
            )

            mtl = chunk["mtl"]
            if "map_Kd" in mtl:
                texture = Texture.load(mtl["map_Kd"])
            else:
                texture = None

            self.vlists.append(vlist)
            self.textures.append(texture)

    def _load_mtl(self, model_file):
        model_dir, file_name = os.path.split(model_file)

        # Create a default material for the model
        default_mtl = {
            "Kd": np.array([1, 1, 1]),
        }

        # Determine the default texture path for the default material
        tex_name = file_name.split(".")[0]
        tex_path = get_file_path("meshes", tex_name, "png")
        if os.path.exists(tex_path):
            default_mtl["map_Kd"] = tex_path

        materials = {"": default_mtl}

        mtl_path = model_file.split(".")[0] + ".mtl"

        if not os.path.exists(mtl_path):
            return materials

        # print('Loading materials from "%s"' % mtl_path)

        mtl_file = open(mtl_path)

        cur_mtl = None

        # For each line of the input file
        for line in mtl_file:
            line = line.rstrip(" \r\n")

            # Skip comments
            if line.startswith("#") or line == "":
                continue

            tokens = line.split(" ")
            tokens = map(lambda t: t.strip(" "), tokens)
            tokens = list(filter(lambda t: t != "", tokens))

            prefix = tokens[0]
            tokens = tokens[1:]

            if prefix == "newmtl":
                cur_mtl = {}
                materials[tokens[0]] = cur_mtl

            # Diffuse color
            if prefix == "Kd":
                vals = list(map(lambda v: float(v), tokens))
                vals = np.array(vals)
                cur_mtl["Kd"] = vals

            # Texture file name
            if prefix == "map_Kd":
                tex_file = tokens[-1]
                tex_file = os.path.join(model_dir, tex_file)
                cur_mtl["map_Kd"] = tex_file

        mtl_file.close()

        return materials

    def render(self):
        for idx, vlist in enumerate(self.vlists):
            texture = self.textures[idx]

            if texture:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(texture.target, texture.id)
            else:
                glDisable(GL_TEXTURE_2D)

            vlist.draw(GL_TRIANGLES)

        glDisable(GL_TEXTURE_2D)
