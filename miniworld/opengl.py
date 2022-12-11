import os
from ctypes import POINTER, byref

import numpy as np
import pyglet

# Solution to https://github.com/maximecb/gym-miniworld/issues/24
# until pyglet support egl officially
from pyglet.gl import (
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_COMPONENT,
    GL_DEPTH_COMPONENT16,
    GL_DEPTH_TEST,
    GL_DRAW_FRAMEBUFFER,
    GL_FLOAT,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT,
    GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER,
    GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS,
    GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT,
    GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE,
    GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER,
    GL_FRAMEBUFFER_UNDEFINED,
    GL_FRAMEBUFFER_UNSUPPORTED,
    GL_GENERATE_MIPMAP_HINT,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_LINES,
    GL_MULTISAMPLE,
    GL_NEAREST,
    GL_NICEST,
    GL_PACK_ALIGNMENT,
    GL_QUADS,
    GL_READ_FRAMEBUFFER,
    GL_RENDERBUFFER,
    GL_RGB,
    GL_RGBA,
    GL_RGBA32F,
    GL_TEXTURE_2D,
    GL_TEXTURE_2D_MULTISAMPLE,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    GL_UNSIGNED_SHORT,
    GLint,
    GLubyte,
    GLuint,
    GLushort,
    gl_info,
    glBegin,
    glBindFramebuffer,
    glBindRenderbuffer,
    glBindTexture,
    glBlitFramebuffer,
    glCheckFramebufferStatus,
    glColor3f,
    glEnable,
    glEnd,
    glFramebufferRenderbuffer,
    glFramebufferTexture2D,
    glGenerateMipmap,
    glGenFramebuffers,
    glGenRenderbuffers,
    glGenTextures,
    glGetIntegerv,
    glHint,
    glNormal3f,
    glPixelStorei,
    glReadPixels,
    glRenderbufferStorage,
    glRenderbufferStorageMultisample,
    glTexImage2D,
    glTexImage2DMultisample,
    glTexParameteri,
    glVertex3f,
    glViewport,
)

from miniworld.utils import get_file_path

if os.environ.get("PYOPENGL_PLATFORM", None) == "egl":
    pyglet.options["headless"] = True


# Mapping of frame buffer error enums to strings
FB_ERROR_ENUMS = {
    GL_FRAMEBUFFER_UNDEFINED: "GL_FRAMEBUFFER_UNDEFINED",
    GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT",
    GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT",
    GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER: "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER",
    GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER: "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER",
    GL_FRAMEBUFFER_UNSUPPORTED: "GL_FRAMEBUFFER_UNSUPPORTED",
    GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE",
    GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS: "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS",
}


class Texture:
    """
    Manage the loading and caching of textures, as well as texture randomization
    """

    # List of textures available for a given path
    tex_paths = {}

    # Cache of textures
    tex_cache = {}

    @classmethod
    def get(self, tex_name, rng=None):
        """
        Load a texture by name (or used a cached version)
        Also performs domain randomization if multiple versions are available.
        """

        paths = self.tex_paths.get(tex_name, [])

        # Get an inventory of the existing texture files
        if len(paths) == 0:
            for i in range(1, 10):
                path = get_file_path("textures", "%s_%d" % (tex_name, i), "png")

                if not os.path.exists(path):
                    break
                paths.append(path)

        assert len(paths) > 0, ValueError(
            'failed to load textures for name "%s"' % tex_name
        )

        # If domain-randomization is to be used
        if rng:
            path_idx = rng.integers(0, len(paths))
            path = paths[path_idx]
        else:
            path = paths[0]

        if path not in self.tex_cache:
            self.tex_cache[path] = Texture(Texture.load(path), tex_name)

        return self.tex_cache[path]

    @classmethod
    def load(cls, tex_path):
        """
        Load a texture based on its path. No domain randomization.
        In most cases, this method should not be used directly.
        """

        # print('Loading texture "%s"' % tex_path)

        img = pyglet.image.load(tex_path)
        tex = img.get_texture()
        glEnable(tex.target)
        glBindTexture(tex.target, tex.id)

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            img.width,
            img.height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            img.get_image_data().get_data("RGBA", img.width * 4),
        )

        # Generate mipmaps (multiple levels of detail)
        glHint(GL_GENERATE_MIPMAP_HINT, GL_NICEST)
        glGenerateMipmap(GL_TEXTURE_2D)

        # Trilinear texture filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

        # Unbind the texture
        glBindTexture(GL_TEXTURE_2D, 0)

        return tex

    def __init__(self, tex, tex_name):
        assert not isinstance(tex, str)
        self.tex = tex
        self.width = self.tex.width
        self.height = self.tex.height
        self.name = tex_name

    def bind(self):
        glBindTexture(self.tex.target, self.tex.id)


class FrameBuffer:
    """
    Manage frame buffers for rendering
    """

    def __init__(self, width, height, num_samples=1):
        """Create the frame buffer objects"""

        assert num_samples > 0
        assert num_samples <= 16

        self.width = width
        self.height = height

        # Create a frame buffer (rendering target)
        self.multi_fbo = GLuint(0)
        glGenFramebuffers(1, byref(self.multi_fbo))
        glBindFramebuffer(GL_FRAMEBUFFER, self.multi_fbo)

        # The try block here is because some OpenGL drivers
        # (Intel GPU drivers on MacBooks in particular) do not
        # support multisampling on frame buffer objects
        try:
            # Ensure that the correct extension is supported
            assert gl_info.have_extension("GL_EXT_framebuffer_multisample")

            # Get the maximum number of samples supported
            MAX_SAMPLES_EXT = 0x8D57
            max_samples = GLint()
            glGetIntegerv(MAX_SAMPLES_EXT, max_samples)
            max_samples = max_samples.value

            if num_samples > max_samples:
                print(f"Falling back to num_samples={max_samples}")
                num_samples = max_samples

            # Create a multisampled texture to render into
            fbTex = GLuint(0)
            glGenTextures(1, byref(fbTex))
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, fbTex)
            glTexImage2DMultisample(
                GL_TEXTURE_2D_MULTISAMPLE, num_samples, GL_RGBA32F, width, height, True
            )
            glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D_MULTISAMPLE,
                fbTex,
                0,
            )

            # Attach a multisampled depth buffer to the FBO
            depth_rb = GLuint(0)
            glGenRenderbuffers(1, byref(depth_rb))
            glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
            glRenderbufferStorageMultisample(
                GL_RENDERBUFFER, num_samples, GL_DEPTH_COMPONENT16, width, height
            )
            glFramebufferRenderbuffer(
                GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb
            )

            # Check that the frame buffer creation succeeded
            res = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            assert res == GL_FRAMEBUFFER_COMPLETE, FB_ERROR_ENUMS.get(res, res)

        except Exception:
            print("Falling back to non-multisampled frame buffer")

            # Create a plain texture to render into
            fbTex = GLuint(0)
            glGenTextures(1, byref(fbTex))
            glBindTexture(GL_TEXTURE_2D, fbTex)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, None
            )
            glFramebufferTexture2D(
                GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbTex, 0
            )

            # Attach depth buffer to FBO
            depth_rb = GLuint(0)
            glGenRenderbuffers(1, byref(depth_rb))
            glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width, height)
            glFramebufferRenderbuffer(
                GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb
            )

        # Sanity check
        res = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        assert res == GL_FRAMEBUFFER_COMPLETE, FB_ERROR_ENUMS.get(res, res)

        # Create the frame buffer used to resolve the final render
        self.final_fbo = GLuint(0)
        glGenFramebuffers(1, byref(self.final_fbo))
        glBindFramebuffer(GL_FRAMEBUFFER, self.final_fbo)

        # Create the texture used to resolve the final render
        fbTex = GLuint(0)
        glGenTextures(1, byref(fbTex))
        glBindTexture(GL_TEXTURE_2D, fbTex)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, None
        )
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbTex, 0
        )

        # Create a depth buffer for the final frame buffer
        depth_rb = GLuint(0)
        glGenRenderbuffers(1, byref(depth_rb))
        glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, width, height)
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb
        )

        # Sanity check
        res = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        assert res == GL_FRAMEBUFFER_COMPLETE, FB_ERROR_ENUMS.get(res, res)

        # Enable depth testing
        glEnable(GL_DEPTH_TEST)

        # Unbind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Array to render the image into (for observation rendering)
        # The array is stored in column-major order
        self.img_array = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    def bind(self):
        """
        Bind the frame buffer before rendering into it
        """

        # Bind the multisampled frame buffer
        glEnable(GL_MULTISAMPLE)
        glBindFramebuffer(GL_FRAMEBUFFER, self.multi_fbo)
        glViewport(0, 0, self.width, self.height)

    def resolve(self):
        """
        Produce a numpy image array from the rendered image
        """

        # Resolve the multisampled frame buffer into the final frame buffer
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.multi_fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.final_fbo)
        glBlitFramebuffer(
            0,
            0,
            self.width,
            self.height,
            0,
            0,
            self.width,
            self.height,
            GL_COLOR_BUFFER_BIT,
            GL_LINEAR,
        )

        # Resolve the depth component as well
        glBlitFramebuffer(
            0,
            0,
            self.width,
            self.height,
            0,
            0,
            self.width,
            self.height,
            GL_DEPTH_BUFFER_BIT,
            GL_NEAREST,
        )

        # Copy the frame buffer contents into a numpy array
        # Note: glReadPixels reads starting from the lower left corner
        glBindFramebuffer(GL_FRAMEBUFFER, self.final_fbo)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glReadPixels(
            0,
            0,
            self.width,
            self.height,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            self.img_array.ctypes.data_as(POINTER(GLubyte)),
        )

        # Unbind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Flip the image because OpenGL maps (0,0) to the lower-left corner
        # Note: this is necessary for gym.wrappers.Monitor to record videos
        # properly, otherwise they are vertically inverted.
        # Note: ascontiguousarray operates in constant time because it
        # does not copy the data
        img = np.ascontiguousarray(np.flip(self.img_array, axis=0))

        return img

    def get_depth_map(self, z_near=0.04, z_far=1.0):
        """
        Read the depth buffer into a depth map
        The values returned are real-world z-distance from the observer
        """

        depth_map = np.zeros(shape=(self.height, self.width, 1), dtype=np.uint16)

        glBindFramebuffer(GL_FRAMEBUFFER, self.final_fbo)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glReadPixels(
            0,
            0,
            self.width,
            self.height,
            GL_DEPTH_COMPONENT,
            GL_UNSIGNED_SHORT,
            depth_map.ctypes.data_as(POINTER(GLushort)),
        )

        # Unbind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Flip the depth map vertically to map OpenAI gym conventions
        depth_map = np.flip(depth_map, axis=0)

        # Transform into floating-point values
        depth_map = depth_map.astype(np.float32) / 65535

        # Convert to real-world z-distances
        clip_z = (depth_map - 0.5) * 2.0
        world_z = -2 * z_far * z_near / (clip_z * (z_far - z_near) - (z_far + z_near))

        depth_map = np.ascontiguousarray(world_z)

        return depth_map


def drawAxes(len=0.1):
    """
    Draw X/Y/Z axes in red/green/blue colors
    """

    glBegin(GL_LINES)

    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(len, 0, 0)

    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, len, 0)

    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, len)

    glEnd()


def drawBox(x_min, x_max, y_min, y_max, z_min, z_max):
    """
    Draw a 3D box
    """

    glBegin(GL_QUADS)

    glNormal3f(0, 0, 1)
    glVertex3f(x_max, y_max, z_max)
    glVertex3f(x_min, y_max, z_max)
    glVertex3f(x_min, y_min, z_max)
    glVertex3f(x_max, y_min, z_max)

    glNormal3f(0, 0, -1)
    glVertex3f(x_min, y_max, z_min)
    glVertex3f(x_max, y_max, z_min)
    glVertex3f(x_max, y_min, z_min)
    glVertex3f(x_min, y_min, z_min)

    glNormal3f(-1, 0, 0)
    glVertex3f(x_min, y_max, z_max)
    glVertex3f(x_min, y_max, z_min)
    glVertex3f(x_min, y_min, z_min)
    glVertex3f(x_min, y_min, z_max)

    glNormal3f(1, 0, 0)
    glVertex3f(x_max, y_max, z_min)
    glVertex3f(x_max, y_max, z_max)
    glVertex3f(x_max, y_min, z_max)
    glVertex3f(x_max, y_min, z_min)

    glNormal3f(0, 1, 0)
    glVertex3f(x_max, y_max, z_max)
    glVertex3f(x_max, y_max, z_min)
    glVertex3f(x_min, y_max, z_min)
    glVertex3f(x_min, y_max, z_max)

    glNormal3f(0, -1, 0)
    glVertex3f(x_max, y_min, z_min)
    glVertex3f(x_max, y_min, z_max)
    glVertex3f(x_min, y_min, z_max)
    glVertex3f(x_min, y_min, z_min)

    glEnd()
