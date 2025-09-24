from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/lasot'
    settings.lasotlang_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/lasot'
    settings.network_path = '/home/SSY/PythonProject/Tracking/SUTrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/nfs'
    settings.otb_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/OTB2015'
    settings.otblang_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/otb_lang'
    settings.prj_dir = '/home/SSY/PythonProject/Tracking/SUTrack'
    settings.result_plot_path = '/home/SSY/PythonProject/Tracking/SUTrack/test/result_plots'
    settings.results_path = '/home/SSY/PythonProject/Tracking/SUTrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/SSY/PythonProject/Tracking/SUTrack'
    settings.segmentation_path = '/home/SSY/PythonProject/Tracking/SUTrack/test/segmentation_results'
    settings.tc128_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/trackingnet'
    settings.uav_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/UAV123'
    settings.vot_path = '/home/SSY/PythonProject/Tracking/SUTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

