# Goals for project in current phase

## Initialize Weights in a seperate methods

The weights are initialized uniformly during the first call to generate, but it would be cleaner to handle this in the constructor (__init__) or via a separate method. This avoids implicit behavior and makes it easier to debug.

## Better Device Management

The device for the ensemble is dynamically assigned in generate if not provided earlier. Ensure self.device is consistently set across all components (models, tokenizers, mappings) to avoid potential mismatches.

