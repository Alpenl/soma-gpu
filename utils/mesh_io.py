import numpy as np
import os
from struct import pack, unpack


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm

def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces


def save_obj_mesh(mesh_path, verts, faces, norms=None, face_normals=None, uvs=None, face_uvs=None):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))

    # write normals
    if norms is not None:
        for n in norms:
            file.write('vn %.4f %.4f %.4f\n' % (n[0], n[1], n[2]))

    if uvs is not None:
        for vt in uvs:
            file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    faces = np.copy(faces) + 1
    if face_normals is not None:
        face_normals = np.copy(face_normals) + 1
    if face_uvs is not None:
        face_uvs = np.copy(face_uvs) + 1

    # write faces
    for fi, f in enumerate(faces):

        if face_normals is None and face_uvs is None:
            file.write('f %d %d %d\n' % (f[0], f[1], f[2]))
        elif face_normals is None and face_uvs is not None:
            file.write('f %d/%d %d/%d %d/%d\n' % (f[0], face_uvs[fi][0], f[1], face_uvs[fi][1], f[2], face_uvs[fi][2]))
        elif face_normals is not None and face_uvs is not None:
            file.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (f[0], face_uvs[fi][0], face_normals[fi][0],
                                                           f[1], face_uvs[fi][1], face_normals[fi][1],
                                                           f[2], face_uvs[fi][2], face_normals[fi][2]))
        else:
            raise ValueError('This cannot happen!')
    file.close()

def readPC2(file, float16=False):
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    data = {}
    bytes = 2 if float16 else 4
    dtype = np.float16 if float16 else np.float32
    with open(file, "rb") as f:
        # Header
        data["sign"] = f.read(12)
        # data['version'] = int.from_bytes(f.read(4), 'little')
        data["version"] = unpack("<i", f.read(4))[0]
        # Num points
        # data['nPoints'] = int.from_bytes(f.read(4), 'little')
        data["nPoints"] = unpack("<i", f.read(4))[0]
        # Start frame
        data["startFrame"] = unpack("f", f.read(4))
        # Sample rate
        data["sampleRate"] = unpack("f", f.read(4))
        # Number of samples
        # data['nSamples'] = int.from_bytes(f.read(4), 'little')
        data["nSamples"] = unpack("<i", f.read(4))[0]
        # Animation data
        size = data["nPoints"] * data["nSamples"] * 3 * bytes
        data["V"] = np.frombuffer(f.read(size), dtype=dtype).astype(np.float32)
        data["V"] = data["V"].reshape(data["nSamples"], data["nPoints"], 3)

    return data


"""
Reads an specific frame of PC2/PC16 files
Inputs:
- file: path to .pc2/.pc16 file
- frame: number of the frame to read
- float16: False for PC2 files, True for PC16
Output:
- T: mesh vertex data at specified frame
"""


def readPC2Frame(file, frame, float16=False):
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    assert frame >= 0 and isinstance(frame, int), "Frame must be a positive integer"
    bytes = 2 if float16 else 4
    dtype = np.float16 if float16 else np.float32
    with open(file, "rb") as f:
        # Num points
        f.seek(16)
        # nPoints = int.from_bytes(f.read(4), 'little')
        nPoints = unpack("<i", f.read(4))[0]
        # Number of samples
        f.seek(28)
        # nSamples = int.from_bytes(f.read(4), 'little')
        nSamples = unpack("<i", f.read(4))[0]
        if frame > nSamples:
            print("Frame index outside size")
            print("\tN. frame: " + str(frame))
            print("\tN. samples: " + str(nSamples))
            return
        # Read frame
        size = nPoints * 3 * bytes
        f.seek(size * frame, 1)  # offset from current '1'
        T = np.frombuffer(f.read(size), dtype=dtype).astype(np.float32)
    return T.reshape(nPoints, 3)


"""
Writes PC2 and PC16 files
Inputs:
- file: path to file (overwrites if exists)
- V: 3D animation data as a three dimensional array (N. Frames x N. Vertices x 3)
- float16: False for writing as PC2 file, True for PC16
This function assumes 'startFrame' to be 0 and 'sampleRate' to be 1
NOTE: 16-bit floats lose precision with high values (positive or negative),
	  we do not recommend using this format for data outside range [-2, 2]
"""


def writePC2(file, V, float16=False):
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    if float16:
        V = V.astype(np.float16)
    else:
        V = V.astype(np.float32)
    with open(file, "wb") as f:
        # Create the header
        headerFormat = "<12siiffi"
        headerStr = pack(
            headerFormat, b"POINTCACHE2\0", 1, V.shape[1], 0, 1, V.shape[0]
        )
        f.write(headerStr)
        # Write vertices
        f.write(V.tobytes())


"""
Appends frames to PC2 and PC16 files
Inputs:
- file: path to file
- V: 3D animation data as a three dimensional array (N. New Frames x N. Vertices x 3)
- float16: False for writing as PC2 file, True for PC16
This function assumes 'startFrame' to be 0 and 'sampleRate' to be 1
NOTE: 16-bit floats lose precision with high values (positive or negative),
	  we do not recommend using this format for data outside range [-2, 2]
"""


def writePC2Frames(file, V, float16=False):
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    # Read file metadata (dimensions)
    if os.path.isfile(file):
        if float16:
            V = V.astype(np.float16)
        else:
            V = V.astype(np.float32)
        with open(file, "rb+") as f:
            # Num points
            f.seek(16)
            nPoints = unpack("<i", f.read(4))[0]
            assert len(V.shape) == 3 and V.shape[1] == nPoints, (
                "Inconsistent dimensions: "
                + str(V.shape)
                + " and should be (-1,"
                + str(nPoints)
                + ",3)"
            )
            # Read n. of samples
            f.seek(28)
            nSamples = unpack("<i", f.read(4))[0]
            # Update n. of samples
            nSamples += V.shape[0]
            f.seek(28)
            f.write(pack("i", nSamples))
            # Append new frame/s
            f.seek(0, 2)
            f.write(V.tobytes())
    else:
        writePC2(file, V, float16)