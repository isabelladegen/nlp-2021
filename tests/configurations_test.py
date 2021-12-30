from hamcrest import *
from src.configurations import *
from dataclasses import fields


def test_default_values_for_all_configurations():
    configs = Configuration()

    assert_that(len(fields(configs)), equal_to(19))  # also will hint to update tests
    assert_that(configs.dataset_name, equal_to('doc2dial'))
    assert_that(configs.document_data_name, equal_to('document_domain'))
    assert_that(configs.rc_data, equal_to('doc2dial_rc'))
    assert_that(configs.data_ignore_verifications, equal_to(True))
    assert_that(configs.data_cache_dir, equal_to('./data_cache_src'))
    assert_that(configs.keep_in_memory, equal_to(True))
    assert_that(configs.window, equal_to(4))


def test_returns_dictionary_for_all_configurations():
    configs = Configuration()

    config_as_dict = configs.as_dict()

    assert_that(len(config_as_dict), equal_to(len(fields(configs))))
    assert_that(config_as_dict['dataset_name'], equal_to('doc2dial'))
    assert_that(config_as_dict['document_data_name'], equal_to('document_domain'))
    assert_that(config_as_dict['rc_data'], equal_to('doc2dial_rc'))
    assert_that(config_as_dict['data_ignore_verifications'], equal_to(True))
    assert_that(config_as_dict['data_cache_dir'], equal_to('./data_cache_src'))
    assert_that(config_as_dict['keep_in_memory'], equal_to(True))
    assert_that(config_as_dict['window'], equal_to(4))
    assert_that(config_as_dict['min_count'], equal_to(1))
    assert_that(config_as_dict['workers'], equal_to(4))


def test_creates_configurations_from_dictionary_different_to_defaults():
    config_dictionary = {'dataset_name': 'some_name', 'data_ignore_verifications': False,
                         'data_cache_dir': 'some_dir', 'keep_in_memory': False,
                         'document_data_name': 'some_data', 'rc_data': 'some_rc_data', 'vector_size': 1000,
                         'window': 300, 'min_count': 400, 'workers': 400, 'dm': 300, 'epochs': 700}

    configs = Configuration(**config_dictionary)

    assert_that(configs.dataset_name, equal_to('some_name'))
    assert_that(configs.document_data_name, equal_to('some_data'))
    assert_that(configs.rc_data, equal_to('some_rc_data'))
    assert_that(configs.data_ignore_verifications, equal_to(False))
    assert_that(configs.data_cache_dir, equal_to('some_dir'))
    assert_that(configs.keep_in_memory, equal_to(False))
    assert_that(configs.vector_size, equal_to(1000))
    assert_that(configs.window, equal_to(300))
    assert_that(configs.min_count, equal_to(400))
    assert_that(configs.workers, equal_to(400))
    assert_that(configs.dm, equal_to(300))
    assert_that(configs.epochs, equal_to(700))


def tests_uses_default_values_if_not_all_values_are_specified():
    config_dictionary = {'dataset_name': 'any', 'keep_in_memory': False, 'epochs': 700}

    configs = Configuration(**config_dictionary)

    assert_that(configs.dataset_name, equal_to('any'))
    assert_that(configs.document_data_name, equal_to('document_domain'))
    assert_that(configs.rc_data, equal_to('doc2dial_rc'))
    assert_that(configs.data_ignore_verifications, equal_to(True))
    assert_that(configs.data_cache_dir, equal_to('./data_cache_src'))
    assert_that(configs.keep_in_memory, equal_to(False))
    assert_that(configs.window, equal_to(4))
    assert_that(configs.min_count, equal_to(1))
    assert_that(configs.workers, equal_to(4))
