from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst.directives.images import Image


class SubstCodeBlock(Directive):
    """
    Usage:
      .. subst-code-block:: pycon

         >>> ...
         '|release|'
    Replaces |release| and |version| using Sphinx config, then emits a highlighted literal_block.
    """
    has_content = True
    required_arguments = 1  # language, e.g. pycon

    def run(self):
        env = getattr(self.state.document.settings, "env", None)

        text = "\n".join(self.content)

        # Replace Sphinx built-ins explicitly
        if env is not None:
            release = getattr(env.config, "release", "")
            version = getattr(env.config, "version", "")
            text = text.replace("|release|", release)
            text = text.replace("|version|", version)

        literal = nodes.literal_block(text, text)
        literal["language"] = self.arguments[0]
        return [literal]
    
class ImageRelease(Image):
    """
    Usage:
      .. image-release:: https://img.shields.io/badge/Version-|release|-blue.svg
         :target: https://pypi.org/project/pyafv
         :alt: PyPI

    Replaces |release| and |version| using Sphinx config, then behaves like .. image::.
    """
    def run(self):
        env = getattr(self.state.document.settings, "env", None)

        if env is not None:
            release = getattr(env.config, "release", "")
            version = getattr(env.config, "version", "")

            # Replace in the required URI argument
            if self.arguments:
                self.arguments[0] = (
                    self.arguments[0]
                    .replace("|release|", release)
                    .replace("|version|", version)
                )

            # Replace in common string options, if present
            for opt in ("target", "alt"):
                if opt in self.options and isinstance(self.options[opt], str):
                    self.options[opt] = (
                        self.options[opt]
                        .replace("|release|", release)
                        .replace("|version|", version)
                    )

        return super().run()

def setup(app):
    app.add_directive("subst-code-block", SubstCodeBlock)
    app.add_directive("image-release", ImageRelease)
    return {"version": "0.1.0", "parallel_read_safe": True, "parallel_write_safe": True}
