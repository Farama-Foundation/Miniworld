import math
import os
import numpy as np
import pyglet
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
    def __init__(self):
        pass



def create_frame_buffers(width, height, num_samples):
    """Create the frame buffer objects"""

    # Create a frame buffer (rendering target)
    multi_fbo = GLuint(0)
    glGenFramebuffers(1, byref(multi_fbo))
    glBindFramebuffer(GL_FRAMEBUFFER, multi_fbo)

    # The try block here is because some OpenGL drivers
    # (Intel GPU drivers on macbooks in particular) do not
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
    final_fbo = GLuint(0)
    glGenFramebuffers(1, byref(final_fbo))
    glBindFramebuffer(GL_FRAMEBUFFER, final_fbo)

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

    return multi_fbo, final_fbo

# TODO: method for resolving MSAA FBO into final FBO
# TODO: method to get rgb array from FBO
# IDEA: possibly, we should allocate rgb arrays for FBOs here too
