# Model Discovery

## Model ID
All models are formed of their artefacts - such as model weights and parameters - and their code.
A `Model ID` contains a hash of the artefacts, and Version Control Information pointing to the model code.

```text
Model ID
Artefact Schema ID: V1:16b155b187b60b211b1b31b118b107b152b185b23b20b152b33b51b10b89b82b154b51b118b232b143b6b120b24b234b72b95b189b84b96:407976
VCS Information: (https://github.com/shareableai/jackdaw, 'develop', bfbfe2cb359e0ac28833fe943aea657a4732e15d)
```

### Artefact Schema ID
All Models have a unique ID, which is taken from the parameters stored on that model. This is known as the `Artefact Schema ID`.

If we had two Tensorflow models, we could say that two models have the same `Artefact Schema ID` if they have
exactly the same layers, with exactly the same parameters.

### VCS Information - Optional
VCS Information links the model's artefacts back to the code responsible for the model. While this
is optional, linking the model directly to its hash and URL allows for Continuous Integration
systems to test model artefacts on deployment.

## Search - Corvus
Corvus is the ShareableAI tool to find models by their name, parameters, Git Branch, etc.