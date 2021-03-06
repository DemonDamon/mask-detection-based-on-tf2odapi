# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for movinet.py."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.projects.movinet.modeling import movinet


class MoViNetTest(parameterized.TestCase, tf.test.TestCase):

  def test_network_creation(self):
    """Test creation of MoViNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')

    network = movinet.Movinet(
        model_id='a0',
        causal=True,
    )
    inputs = tf.keras.Input(shape=(8, 128, 128, 3), batch_size=1)
    endpoints, states = network(inputs)

    self.assertAllEqual(endpoints['stem'].shape, [1, 8, 64, 64, 8])
    self.assertAllEqual(endpoints['b0/l0'].shape, [1, 8, 32, 32, 8])
    self.assertAllEqual(endpoints['b1/l0'].shape, [1, 8, 16, 16, 32])
    self.assertAllEqual(endpoints['b2/l0'].shape, [1, 8, 8, 8, 56])
    self.assertAllEqual(endpoints['b3/l0'].shape, [1, 8, 8, 8, 56])
    self.assertAllEqual(endpoints['b4/l0'].shape, [1, 8, 4, 4, 104])
    self.assertAllEqual(endpoints['head'].shape, [1, 1, 1, 1, 480])

    self.assertNotEmpty(states)

  def test_network_with_states(self):
    """Test creation of MoViNet family models with states."""
    tf.keras.backend.set_image_data_format('channels_last')

    network = movinet.Movinet(
        model_id='a0',
        causal=True,
    )
    inputs = tf.ones([1, 8, 128, 128, 3])

    _, states = network(inputs)
    endpoints, new_states = network(dict(image=inputs, states=states))

    self.assertAllEqual(endpoints['stem'].shape, [1, 8, 64, 64, 8])
    self.assertAllEqual(endpoints['b0/l0'].shape, [1, 8, 32, 32, 8])
    self.assertAllEqual(endpoints['b1/l0'].shape, [1, 8, 16, 16, 32])
    self.assertAllEqual(endpoints['b2/l0'].shape, [1, 8, 8, 8, 56])
    self.assertAllEqual(endpoints['b3/l0'].shape, [1, 8, 8, 8, 56])
    self.assertAllEqual(endpoints['b4/l0'].shape, [1, 8, 4, 4, 104])
    self.assertAllEqual(endpoints['head'].shape, [1, 1, 1, 1, 480])

    self.assertNotEmpty(states)
    self.assertNotEmpty(new_states)

  def test_movinet_stream(self):
    tf.keras.backend.set_image_data_format('channels_last')

    model = movinet.Movinet(
        model_id='a0',
        causal=True,
    )
    inputs = tf.ones([1, 5, 128, 128, 3])

    expected_endpoints, _ = model(dict(image=inputs, states={}))

    frames = tf.split(inputs, inputs.shape[1], axis=1)

    output, states = None, {}
    for frame in frames:
      output, states = model(dict(image=frame, states=states))
    predicted_endpoints = output

    predicted = predicted_endpoints['head']

    # The expected final output is simply the mean across frames
    expected = expected_endpoints['head']
    expected = tf.reduce_mean(expected, 1, keepdims=True)

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected, 1e-5, 1e-5)

  def test_movinet_2plus1d_stream(self):
    tf.keras.backend.set_image_data_format('channels_last')

    model = movinet.Movinet(
        model_id='a0',
        causal=True,
        conv_type='2plus1d',
    )
    inputs = tf.ones([1, 5, 128, 128, 3])

    expected_endpoints, _ = model(dict(image=inputs, states={}))

    frames = tf.split(inputs, inputs.shape[1], axis=1)

    output, states = None, {}
    for frame in frames:
      output, states = model(dict(image=frame, states=states))
    predicted_endpoints = output

    predicted = predicted_endpoints['head']

    # The expected final output is simply the mean across frames
    expected = expected_endpoints['head']
    expected = tf.reduce_mean(expected, 1, keepdims=True)

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected, 1e-5, 1e-5)

  def test_movinet_3d_2plus1d_stream(self):
    tf.keras.backend.set_image_data_format('channels_last')

    model = movinet.Movinet(
        model_id='a0',
        causal=True,
        conv_type='3d_2plus1d',
    )
    inputs = tf.ones([1, 5, 128, 128, 3])

    expected_endpoints, _ = model(dict(image=inputs, states={}))

    frames = tf.split(inputs, inputs.shape[1], axis=1)

    output, states = None, {}
    for frame in frames:
      output, states = model(dict(image=frame, states=states))
    predicted_endpoints = output

    predicted = predicted_endpoints['head']

    # The expected final output is simply the mean across frames
    expected = expected_endpoints['head']
    expected = tf.reduce_mean(expected, 1, keepdims=True)

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected, 1e-5, 1e-5)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id='a0',
        causal=True,
        use_positional_encoding=True,
    )
    network = movinet.Movinet(**kwargs)

    # Create another network object from the first object's config.
    new_network = movinet.Movinet.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
