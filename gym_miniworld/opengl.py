import math
import os
import numpy as np
import pyglet
from pyglet.gl import *
from ctypes import byref, POINTER
from .utils import *

class Texture:
    """
    Manage the caching of textures, and texture randomization
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
                path = get_file_path('textures', '%s_%d' % (tex_name, i), 'png')
                if not os.path.exists(path):
                    break
                paths.append(path)

        assert len(paths) > 0, 'failed to load textures for name "%s"' % tex_name

        if rng:
            path_idx = rng.int(0, len(paths))
            path = paths[path_idx]
        else:
            path = paths[0]

        if path not in self.tex_cache:
            self.tex_cache[path] = Texture(load_texture(path))

        return self.tex_cache[path]

    @classmethod
    def load(tex_path):
        """
        Load a texture based on its path. No domain randomization.
        In mose cases, this method should not be used directly.
        """

        print('loading texture "%s"' % tex_path)

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
            img.get_image_data().get_data('RGBA', img.width * 4)
        )

        return tex

    def __init__(self, tex):
        assert not isinstance(tex, str)
        self.tex = tex

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
        self.num_samples = num_samples

        # Create a frame buffer (rendering target)
        self.multi_fbo = GLuint(0)
        glGenFramebuffers(1, byref(self.multi_fbo))
        glBindFramebuffer(GL_FRAMEBUFFER, self.multi_fbo)

        # The try block here is because some OpenGL drivers
        # (Intel GPU drivers on MacBooks in particular) do not
        # support multisampling on frame buffer objects
        try:
            # Create a multisampled texture to render into
            fbTex = GLuint(0)
            glGenTextures( 1, byref(fbTex));
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, fbTex);
            glTexImage2DMultisample(
                GL_TEXTURE_2D_MULTISAMPLE,
                num_samples,
                GL_RGBA32F,
                width,
                height,
                True
            );
            glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D_MULTISAMPLE,
                fbTex,
                0
            );

            # Attach a multisampled depth buffer to the FBO
            depth_rb = GLuint(0)
            glGenRenderbuffers(1, byref(depth_rb))
            glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, num_samples, GL_DEPTH_COMPONENT, width, height);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb);

        except:
            print('Falling back to non-multisampled frame buffer')

            # Create a plain texture texture to render into
            fbTex = GLuint(0)
            glGenTextures( 1, byref(fbTex));
            glBindTexture(GL_TEXTURE_2D, fbTex);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                width,
                height,
                0,
                GL_RGBA,
                GL_FLOAT,
                None
            )
            glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                fbTex,
                0
            );

            # Attach depth buffer to FBO
            depth_rb = GLuint(0)
            glGenRenderbuffers(1, byref(depth_rb))
            glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb);

        # Sanity check
        if pyglet.options['debug_gl']:
            res = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            assert res == GL_FRAMEBUFFER_COMPLETE

        # Create the frame buffer used to resolve the final render
        self.final_fbo = GLuint(0)
        glGenFramebuffers(1, byref(self.final_fbo))
        glBindFramebuffer(GL_FRAMEBUFFER, self.final_fbo)

        # Create the texture used to resolve the final render
        fbTex = GLuint(0)
        glGenTextures(1, byref(fbTex))
        glBindTexture(GL_TEXTURE_2D, fbTex)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            width,
            height,
            0,
            GL_RGBA,
            GL_FLOAT,
            None
        )
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            fbTex,
            0
        )
        if pyglet.options['debug_gl']:
          res = glCheckFramebufferStatus(GL_FRAMEBUFFER)
          assert res == GL_FRAMEBUFFER_COMPLETE

        # Enable depth testing
        glEnable(GL_DEPTH_TEST)

        # Unbind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # FIXME: store in Fortran order, order='f' ?
        # The array is stored in column-major order

        # Array to render the image into (for observation rendering)
        self.img_array = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        #print(self.img_array.strides)
        #print(self.img_array.flags)

    def bind(self):
        """
        Bind the frame buffer before rendering into it
        """

        # Bind the multisampled frame buffer
        glEnable(GL_MULTISAMPLE)
        glBindFramebuffer(GL_FRAMEBUFFER, self.multi_fbo);
        glViewport(0, 0, self.width, self.height)

    def resolve(self):
        """
        Produce a numpy image array from the rendered image
        """

        # Resolve the multisampled frame buffer into the final frame buffer
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.multi_fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.final_fbo);
        glBlitFramebuffer(
            0, 0,
            self.width, self.height,
            0, 0,
            self.width, self.height,
            GL_COLOR_BUFFER_BIT,
            GL_LINEAR
        );

        # Copy the frame buffer contents into a numpy array
        # Note: glReadPixels reads starting from the lower left corner
        glBindFramebuffer(GL_FRAMEBUFFER, self.final_fbo);
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glReadPixels(
            0,
            0,
            self.width,
            self.height,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            self.img_array.ctypes.data_as(POINTER(GLubyte))
        )

        # Unbind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        # Flip the image because OpenGL maps (0,0) to the lower-left corner
        # Note: this is necessary for gym.wrappers.Monitor to record videos
        # properly, otherwise they are vertically inverted.
        self.img_array = np.ascontiguousarray(np.flip(self.img_array, axis=0))

        return self.img_array
