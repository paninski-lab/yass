"""
Testing pipeline.run
"""
import util
import yaml


def test_works_with_threshold_config(path_to_threshold_config,
                                     make_tmp_folder):
    from yass import pipeline
    util.seed(0)

    pipeline.run(path_to_threshold_config, output_dir=make_tmp_folder)


def test_works_with_nnet_config(monkeypatch, path_to_nnet_config,
                                make_tmp_folder):
    monkeypatch.setattr('yass.neuralnetwork.KerasModel', util.DummyKerasModel)
    from yass import pipeline
    util.seed(0)

    pipeline.run(path_to_nnet_config, output_dir=make_tmp_folder)


def test_works_with_sample_config_passing_dict(monkeypatch,
                                               path_to_threshold_config,
                                               make_tmp_folder):
    from yass import pipeline
    util.seed(0)

    with open(path_to_threshold_config) as f:
        cfg = yaml.load(f)

    pipeline.run(cfg, output_dir=make_tmp_folder)
