import  numpy
import os
from matplotlib import pylab
# import pandas

sep = os.sep
def distance(list):
    return numpy.sqrt(numpy.sum([m**2 for m in list]))

def export_two_list_file(folder, name, list_x, list_y):
    '''

    :param folder:
    :param name: export in txt file two lists, in columns, list should have same size
    :param list_x:
    :param list_y:
    :return:
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(folder + sep + name + ".txt", "w") as text_file:
        for i in range(0, len(list_x)):
            text_file.write(str(list_x[i]) + "\t" + str(list_y[i]) + "\n")

def export_lists_file(folder, name, list_of_list):
    '''

    :param folder:
    :param name: export in txt file two lists, in columns, list should have same size
    :param list_x:
    :param list_y:
    :return:
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)

    list_0 = list_of_list[0]

    with open(folder + sep + name + ".txt", "w") as text_file:
        for i in range(0, len(list_0)):

            for list in list_of_list:

                text_file.write(str(list[i]) + "\t")
            text_file.write("\n")

def export_cvs(folder, name, list_of_list):
    '''

    :param folder:
    :param name: export in txt file two lists, in columns, list should have same size
    :param list_x:
    :param list_y:
    :return:
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
    list_0 = list_of_list[0]
    with open(folder + sep + name + ".cvs", "w") as text_file:
        for i in range(0, len(list_0)):
            for list in list_of_list:
                text_file.write(str(list[i]) + ";")
            text_file.write("\n") \

def get_results_info(file_name, pandas):
    results = pandas.read_table(file_name)
    return results

def remove_channels(results, list_channels):
    # exlude the channels numbered (0-len(channels))
    # remove cerain channels from results
    results_min_channs = results
    for chnn in list_channels:
        results_min_channs = results_min_channs.query('channel_in_multipoint !=' + str(chnn))

    return results_min_channs

def get_filtered_results(results):
    results_filt = results
    results_filt = results.query('division_age < 2.5')  # divisions lower than 150 min
    results_filt = results_filt.query('division_age > 0.25')  # divisions higher than 15min
    return results_filt

def get_division_events(results_filt):
    '''

      :param results_filt:
      :return: results from the cells that are about to divide
      '''
    division_events = results_filt.query('about_to_divide == 1')
    return division_events

def scatter_division_events(division_events):
    pylab.scatter(division_events.timepoint / (60.0), division_events.division_age * 60)
    pylab.title('Scatter plot of detected division events')
    pylab.ylabel('Division time [min]')
    pylab.xlabel('Experiment time [min]')
    pylab.show()

def get_division_time(division_events):
    """
    Gibes a dataframe of division time (in minutes) of the events contain in the input
    :param division_events: dataframe to take the division time
    :return: dataframe containing the division times (in minutes) of the elements conained in the input dataframe
    """
    division_time = division_events.division_age * 60  # division_age_at_duplication
    return division_time

def division_time_histogram(pylab, division_time):
    division_time.hist(alpha=0.5)
    pylab.title("division_time ")
    pylab.xlabel("division time (min)")
    pylab.show()

def pre_division_cells(results):
    # Dividing cells can be identified by the about_to_divide == 1 flag,
    pre_division = results.query('about_to_divide==1')
    return pre_division

def post_division_cells(results):
    # cells, which resulted from a division have cell_age == 0
    post_division = results.query('cell_age==0')
    return  post_division

def histogram_length_pre_and_pos_division(pre_division, post_division):
    print("Mean length_pre_division \t", pre_division.length.mean())
    print("Mean length_post_dvision \t", post_division.length.mean())

    print("Std length_pre_division \t", pre_division.length.std())
    print("Std length_post_dvision \t", post_division.length.std())

    pylab.title('Cell lengths pre/post-division')
    pylab.xlabel('Length [µm]')
    pylab.ylabel('Occurrence [#]')
    pre_division.length.hist(alpha=0.5, label='Pre-division')
    post_division.length.hist(alpha=0.5, label='Post-division')
    pylab.legend()
    pylab.xlim(0, 10)
    pylab.show()

def print_div_time_stats_(numpy, div_time):
    n_ev = len(div_time)
    division_time_mean = div_time.mean()
    # division_time_median = div_time.median()
    division_time_std = div_time.std()

    print("Division time:")
    print("\t N \t", n_ev)
    print("\t Mean \t", round(division_time_mean, 2), '\tmin')
    # print("\t Median \t", round(division_time_median, 2), '\tmin')
    print("\t Std \t", round(division_time_std, 2), '\tmin')
    print("\t Error \t", round(division_time_std / numpy.sqrt(n_ev), 2), "\tmin")

def print_stats_2(numpy, div_time, results_len):
    n_ev = len(div_time)
    division_time_mean = div_time.mean()
    division_time_median = div_time.median()
    division_time_std = div_time.std()

    n_ev_len = len(results_len)
    print("Division time (min) \t\t\t\t\t Length (µm)")
    print("N\tMean\tMedian\tStd\tError\tMean\tStd\tError\t")
    print(n_ev, "\t",  round(division_time_mean, 2),"\t", division_time_median,"\t", round(division_time_std, 2),"\t", round(division_time_std / numpy.sqrt(n_ev), 2), "\t", round(results_len.mean(), 2),"\t", round(results_len.std(), 2), "\t", round(results_len.std() / numpy.sqrt(n_ev_len), 4))

def print_stats_3(numpy, div_time):
    n_ev = len(div_time)
    division_time_mean = div_time.mean()
    division_time_median = div_time.median()
    division_time_std = div_time.std()


    # print("Division time (min) ")
    # print("N\tMean\tStd\tError")
    print( "\t", n_ev, "\t",  round(division_time_mean, 1),
          "\t", round(division_time_std, 3),
          "\t", round(division_time_std / numpy.sqrt(n_ev), 3))



def print_list_stats(numpy, name, _list):
    n_ev = len(_list)
    _mean = numpy.mean(_list)
    _std = numpy.std(_list)
    _error=_std / numpy.sqrt(n_ev)
    print( name, "\t",n_ev, "\t",round(_mean, 3), "\t",round(_std, 3), "\t",round(_error, 3))
    return n_ev, _mean, _std, _error


def print_stats_4(numpy, name, div_time):
    n_ev = len(div_time)
    division_time_mean = div_time.mean()
    division_time_median = div_time.median()
    division_time_std = div_time.std()

    # print("Division time (min) ")
    # print("N\tMean\tStd\tError")

    print( name, "\t",
           n_ev, "\t",
           round(division_time_mean, 3),
          "\t",
           round(division_time_std, 3),
          "\t",
           round(division_time_std / numpy.sqrt(n_ev), 3))

    return [
        n_ev,
        division_time_mean,
        division_time_std,
        division_time_std / numpy.sqrt(n_ev)
        ]

def print_stats_5(numpy, name, div_time, prnt):
    n_ev = len(div_time)
    division_time_mean = div_time.mean()
    division_time_median = div_time.median()
    division_time_std = div_time.std()

    # print("Division time (min) ")
    # print("N\tMean\tStd\tError")
    if prnt:
        print( name, "\t",
               n_ev, "\t",
               round(division_time_mean, 3),
              "\t",
               round(division_time_std, 3),
              "\t",
               round(division_time_std / numpy.sqrt(n_ev), 3))

    return [
        n_ev,
        round(division_time_mean, 3),
        round(division_time_std, 3),
        round(division_time_std / numpy.sqrt(n_ev), 3)
        ]

def print_length_stats(numpy, results_):
    n_ev = len(results_)

    print("Length:")
    print("\t N \t ", n_ev)
    print("\t Mean \t", round(results_.mean(), 2), "\tµm")
    print("\t Std \t", round(results_.std(), 2), "\tµm")
    print("\t Error \t", round(results_.std() / numpy.sqrt(n_ev), 4), "\tµm")

############################

def get_division_time_per_traces(results_):
    """
    :param results_:
    :return:
    """
    s = results_.groupby("unique_id_on_device").division_age.max()*60

    return s


def get_growth_rate_per_traces(results_):
    """
    :param results_:
    :return:growth rate being ln(2)/doubling time. Units [1/min]
    """
    s = results_.groupby("unique_id_on_device").growth_rate.min()

    return s

def get_groups_by_traces(results_):
    """

    :param results_:
    :return:
    """

    s = results_.groupby("unique_id_on_device")

    return s

def get_average_length_per_traces(results_):
    """

    :param results_:
    :return:
    """
    s = results_.groupby("unique_id_on_device").length.mean()

    return s

def get_average_fluorescence_per_traces(results_, fluorescence):
    """

    :param results_:
    :return:
    """

    if fluorescence == 0:
        s = results_.groupby("unique_id_on_device").fluorescence_raw_0.mean()-500
    if fluorescence == 1:
        s = results_.groupby("unique_id_on_device").fluorescence_raw_1.mean()-500
    return s

def get_simple_elongation_rate_per_traces(results_):
    """
    :param results_:
    :return:

    """
    sub_results_ = results_.query("about_to_divide == 0")#all but the cells that are about to divide
    sub_results_ = sub_results_.query("cell_age != 0")

    grouped = sub_results_.groupby("unique_id_on_device")
    s = (grouped.length.last() - grouped.length.first()) / (grouped.cell_age.last() * 60)
    return s

def get_average_elongation_rate_per_traces(results_):
    """
    :param results_:
    :return:mean elongation rate
    """
    grouped = results_.groupby("unique_id_on_device")
    s = grouped.elongation_rate.mean()
    return s

def get_name(path_file, fov, name):
    file_name = path_file + sep + str(fov) + sep + name
    return file_name

def set_unique_id_on_device(a):
    a["unique_id_on_device"] = a["fov"] + a["position_on_device"] + a["uid_cell"].apply(str)

def set_elongation_rate(a):
    a["elongation_rate"] = a.elongation_rate * 0.16 * 60
    a["elong_rate_over_length"] = a.elongation_rate / a.length

def set_growth_rate(a):

    a["growth_rate"] = numpy.log(2)/a["division_age"]

def length_list_per_channel(results_, ch):
    results_channel = results_.query("channel_in_multipoint == " + str(ch))

    average_length_filt_cleaned = get_average_length_per_traces(results_channel)

    return average_length_filt_cleaned

def elongation_list_per_channel(results_, ch, simple):
    results_channel = results_.query("channel_in_multipoint == " + str(ch))

    if simple:
        average_length_filt_cleaned = get_simple_elongation_rate_per_traces(results_channel)
    else:
        average_length_filt_cleaned = get_average_elongation_rate_per_traces(results_channel)

    return average_length_filt_cleaned


def export_list(name, lista):
    with open(name + ".txt", "w") as text_file:
        for i in lista:
            text_file.write(str(i) + "\n")

##########################################cleaning up events

def remove_short_tracks_and_its_daughters(results, short_time_minutes):
    short_time_hours_theshold = short_time_minutes / 60


    # id of short tracks
    short_tracks = results.query("division_age <= " + str(short_time_hours_theshold))
    short_uid_cell = short_tracks["uid_cell"]

    # identifying daughters of cells with short tracks
    evaluation_short_dt_daughters = results.isin({'uid_parent': short_uid_cell.values})['uid_parent']

    # removing daughters
    subset_wo_short_daughters = results[-evaluation_short_dt_daughters]

    # removing all short tracks
    final_subset = subset_wo_short_daughters.query("division_age > " + str(short_time_hours_theshold + 0.01))


    return final_subset

#WORKING ON
# def remove_traces_w_wrong_fluorescence(results, short_time_minutes, fls_channels):
#     wrong_value_1 = 0
#
#
#     # id of wrong traces
#     for f in fls_channels:
#         wrong_fls_cells = results.query("fluorescence_"+str(f)+" <" + str(wrong_value_1))
#         wrong_uid_cell = wrong_fls_cells["uid_cell"]
#
#     # identifying daughters of cells with short tracks
#     evaluation_short_dt_daughters = results.isin({'uid_parent': short_uid_cell.values})['uid_parent']
#
#     # removing daughters
#     subset_wo_short_daughters = results[-wrong_uid_cell]
#
#     # removing all short tracks
#     final_subset = subset_wo_short_daughters.query("division_age > " + str(short_time_hours_theshold + 0.01))
#
#
#     return final_subset

def remove_large_size_cells_or_filamentus_cells(results, large_size_threshold):

    # id of short tracks


    large_cells = results.query("length >= " + str(large_size_threshold))
    large_uid_cells = large_cells["uid_cell"]

    evaluation_large_cells_daughters = results.isin({'uid_parent': large_uid_cells})['uid_parent']

    # removing daughters
    subset_wo_big_cells_daughters = results[-evaluation_large_cells_daughters]

    # removing all short tracks
    final_subset =  subset_wo_big_cells_daughters.query("length <" + str(large_size_threshold))


    return final_subset


def remove_cells_do_not_divide(results):
    return results.query("division_age > 0")


def remove_spontaneous_cells(results):

    nan = numpy.nan
    results_new = results.query("division_age != @nan")
    return results_new


def remove_mistakes(results, short_time_minutes, large_size_threshold, division_time_max_threshold):

    #remove_cells_do_not_divide
    results_clean = results.query("division_age > 0")

    results_clean = remove_short_tracks_and_its_daughters(results_clean, short_time_minutes)

    results_clean = remove_large_size_cells_or_filamentus_cells(results_clean, large_size_threshold)

    results_clean = remove_short_tracks_and_its_daughters(results_clean, short_time_minutes)

    results_clean = remove_short_tracks_and_its_daughters(results_clean, short_time_minutes)

    # discard first cell per channel
    results_clean = results_clean.query("uid_parent!=0")
    results_clean = remove_spontaneous_cells(results_clean)

    long_time_hours_threshold = division_time_max_threshold / 60
    results_clean = results_clean.query("division_age < " + str(long_time_hours_threshold))

    return results_clean


def remove_mistakes_light(results):
    results_clean = remove_cells_do_not_divide(results)

    # discard first cell per channel
    results_clean = results_clean.query("uid_parent!=0")

    results_clean = remove_spontaneous_cells(results_clean)

    return results_clean


def remove_ugly_channels(results, timepoint_num_max, fov):
    '''
    Discard the events of a complete channel if it was empty at any picture before the picture number 'timepoint_num_max'
    if before a picture (timepoint_num) of the experiment a channel is empty, discard all the channel from the analysis
    :param results:
    :param timepoint_num_max: number of the higher number of picture which taken in account to check if there is empty channels
    :return: filtered results
    '''

    early_results = results.query("timepoint_num < " + str(timepoint_num_max))

    dt_per_channel = early_results.groupby("channel_in_multipoint")

    number_occupied_pics_per_channel = dt_per_channel.timepoint_num.nunique()


    list_channels_to_remove = [indx for indx in number_occupied_pics_per_channel.index if
                          number_occupied_pics_per_channel[indx] < timepoint_num_max]
    # num_channels_remove = len(list_channels_to_remove)

    new_results = remove_channels(results, list_channels_to_remove)

    return new_results


########################################################


def div_time_list_per_channel(results_, ch):
    results_channel = results_.query("channel_in_multipoint == " + str(ch))
    division_events_channel=get_division_events(results_channel)
    division_time_channel=get_division_time(division_events_channel)
    return division_time_channel


def dt_per_traces_list_per_channel(results_, ch):
    results_channel = results_.query("channel_in_multipoint == " + str(ch))

    average_dt_per_traces_filt_cleaned = get_division_time_per_traces(results_channel)

    return average_dt_per_traces_filt_cleaned


def growth_rate_per_traces_list_per_channel(results_, ch):
    results_channel = results_.query("channel_in_multipoint == " + str(ch))

    average_gr_per_traces_filt_cleaned = get_growth_rate_per_traces(results_channel)

    return average_gr_per_traces_filt_cleaned


def fluorescence_per_traces_list_per_channel(results_, cell_channel, fls_channel):
    results_channel = results_.query("channel_in_multipoint == " + str(cell_channel))

    average_fls_per_traces_filt_cleaned = get_average_fluorescence_per_traces(results_channel, fls_channel)

    return average_fls_per_traces_filt_cleaned

###################################

def export_division_time_position_channels( fovs, position, results_filt_cleaned):
    print("~~filtered + cleaned~~")
    print("Division time (min) ")
    print("\tN\tMean\tStd\tError")

    for fov in fovs:
        results_filt_cleaned_ = results_filt_cleaned.query("fov == @fov")

        division_events_filt_cleaned = get_division_events(results_filt_cleaned_)
        division_time_filt_cleaned = get_division_time(division_events_filt_cleaned)

        print_stats_3(numpy, division_time_filt_cleaned)

    for fov in fovs:
        fov_pos = fov + "_" + position

        results_filt_cleaned_ = results_filt_cleaned.query("fov == @fov")

        newpath = r'.\files'

        if not os.path.exists(newpath):
            os.makedirs(newpath)

        nan_ = numpy.nan


        with open(newpath +sep+ fov_pos + "_dt_per_channel_molyso.txt", "w") as text_file:

            for ch in range(0, 29):
                division_time_filt_cleaned = div_time_list_per_channel(results_filt_cleaned_, ch)

                l = len(division_time_filt_cleaned)

                if l != 0:
                    text_file.write(str(ch) + "\t")
                    for a in division_time_filt_cleaned:
                        text_file.write(str(numpy.round(a, 2)) + ",")
                    text_file.write("\n")


def export_dt_per_traces_position_channels( fovs, position, results_filt_cleaned):
    print("~~filtered + cleaned~~")
    print("dt_per_traces (min)")
    print("\tN\tMean\tStd\tError")

    for fov in fovs:
        results_filt_cleaned_ = results_filt_cleaned.query("fov == @fov")

        #####################
        average_dt_per_traces_filt_cleaned = get_division_time_per_traces(results_filt_cleaned_)
        #####################

        print_stats_3(numpy, average_dt_per_traces_filt_cleaned)

    for fov in fovs:
        fov_pos = fov + "_" + position

        results_filt_cleaned_ = results_filt_cleaned.query("fov == @fov")

        newpath = r'.\files'

        if not os.path.exists(newpath):
            os.makedirs(newpath)
        nan_ = numpy.nan
        with open(newpath +sep+ fov_pos + "_dt_per_traces_per_channel_molyso.txt", "w") as text_file:
            for ch in range(0, 29):
                ##################
                dt_traces_filt_cleaned =dt_per_traces_list_per_channel(results_filt_cleaned_, ch)
                ##################

                l = len(dt_traces_filt_cleaned)

                if l != 0:
                    text_file.write(str(ch) + "\t")
                    for a in dt_traces_filt_cleaned:
                        text_file.write(str(numpy.round(a, 2)) + ",")
                    text_file.write("\n")


def export_length_files_position_channels(fovs, position, results_filt_cleaned):
    print("~~filtered + cleaned~~")
    print("Length (um)")
    print("\tN\tMean\tStd\tError")

    for fov in fovs:
        results_filt_cleaned_ = results_filt_cleaned.query("fov == @fov")
        average_length_filt_cleaned = get_average_length_per_traces(results_filt_cleaned_)

        print_stats_3(numpy, average_length_filt_cleaned)

    for fov in fovs:
        fov_pos = fov + "_" + position

        results_filt_cleaned_ = results_filt_cleaned.query("fov == @fov")

        newpath = r'.\files'

        if not os.path.exists(newpath):

            os.makedirs(newpath)
        nan_ = numpy.nan
        with open(newpath +sep+ fov_pos + "_length_per_traces_per_channel_molyso.txt", "w") as text_file:
            for ch in range(0, 29):
                length_filt_cleaned = length_list_per_channel(results_filt_cleaned_, ch)

                l = len(length_filt_cleaned)

                if l != 0:
                    text_file.write(str(ch) + "\t")
                    for a in length_filt_cleaned:
                        text_file.write(str(numpy.round(a, 2)) + ",")
                    text_file.write("\n")


def export_elongation_files_position_channels(fovs, position, results_filt_cleaned, simple,):
    print("~~filtered + cleaned~~")
    print("Elongation (um)")
    print("\tN\tMean\tStd\tError")

    for fov in fovs:
        results_filt_cleaned_ = results_filt_cleaned.query("fov == @fov")
        if simple:
            average_length_filt_cleaned = get_simple_elongation_rate_per_traces(results_filt_cleaned_)
            name = "_elongation_simple_per_traces_per_channel_molyso"
        else:
            average_length_filt_cleaned = get_average_elongation_rate_per_traces(results_filt_cleaned_)
            name = "_elongation_per_traces_per_channel_molyso"


        print_stats_3(numpy, average_length_filt_cleaned)

    for fov in fovs:
        fov_pos = fov + "_" + position

        results_filt_cleaned_ = results_filt_cleaned.query("fov == @fov")
        newpath = r'.\files'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        nan_ = numpy.nan
        with open(newpath + sep + fov_pos + name + ".txt", "w") as text_file:
            for ch in range(0, 29):

                length_filt_cleaned = elongation_list_per_channel(results_filt_cleaned_, ch, simple)


                l = len(length_filt_cleaned)

                if l != 0:
                    text_file.write(str(ch) + "\t")
                    for a in length_filt_cleaned:
                        text_file.write(str(numpy.round(a, 5)) + ",")
                    text_file.write("\n")


def export_growth_rate_per_traces_position_channels( fovs, position, results_filt_cleaned):

    print("gr_per_traces (min)")
    print("\tN\tMean\tStd\tError")

    for fov in fovs:
        results_filt_cleaned_ = results_filt_cleaned.query("fov == @fov")

        #####################
        average_gr_per_traces_filt_cleaned = get_growth_rate_per_traces(results_filt_cleaned_)
        #####################
    #
        print_stats_3(numpy, average_gr_per_traces_filt_cleaned)

    for fov in fovs:
        fov_pos = fov + "_" + position

        results_filt_cleaned_ = results_filt_cleaned.query("fov == @fov")
    #
        newpath = r'.\files'

        if not os.path.exists(newpath):
            os.makedirs(newpath)
        nan_ = numpy.nan
        with open(newpath +sep+ fov_pos + "_growth_rate_per_traces_per_channel_molyso.txt", "w") as text_file:
            for ch in range(0, 29):
                growth_rate_traces_filt_cleaned = growth_rate_per_traces_list_per_channel(results_filt_cleaned_, ch)
                ##################

                l = len(growth_rate_traces_filt_cleaned)
    #
                if l != 0:
                    text_file.write(str(ch) + "\t")
                    for a in growth_rate_traces_filt_cleaned:
                        text_file.write(str(numpy.round(a, 6)) + ",")
                    text_file.write("\n")


def export_fluorescence_per_traces_position_channels(fovs, position, results_filt_cleaned, fluorescence_channels,):

    for channel in fluorescence_channels:
        print("fluorescence_per_traces - Channel " + str(channel))
        print("\tN\tMean\tStd\tError")
        average_fls_per_traces_filt_cleaned = {}
        for fov in fovs:
            results_filt_cleaned_ = results_filt_cleaned.query("fov == @fov")
            average_fls_per_traces_filt_cleaned[channel] = get_average_fluorescence_per_traces(results_filt_cleaned_, channel)
            print_stats_3(numpy, average_fls_per_traces_filt_cleaned[channel])

    for fov in fovs:
        fov_pos = fov + "_" + position

        results_filt_cleaned_ = results_filt_cleaned.query("fov == @fov")
    #
        newpath = r'.\files'

        if not os.path.exists(newpath):
            os.makedirs(newpath)
        nan_ = numpy.nan
        for fls_channel in fluorescence_channels:

            with open(newpath +sep+ fov_pos + "_fluorescence_" + str(fls_channel) + "_per_traces_per_channel_molyso.txt", "w") as text_file:
                for ch in range(0, 29):

                        growth_rate_traces_filt_cleaned = fluorescence_per_traces_list_per_channel(results_filt_cleaned_, ch, fls_channel)
                        ##################

                        l = len(growth_rate_traces_filt_cleaned)
            #
                        if l != 0:
                            text_file.write(str(ch) + "\t")
                            for a in growth_rate_traces_filt_cleaned:
                                text_file.write(str(numpy.round(a, 6)) + ",")
                            text_file.write("\n")


def difference_in_percentage_normalized(mean_L, error_L):
    mean_ = [100 * (m - mean_L[0]) / mean_L[0] for m in mean_L]
    error_ = []

    for x in range(0, len(mean_L)):
        f = mean_L[x] / mean_L[0]
        a = numpy.sqrt((error_L[x] / (mean_L[x])) ** 2 + (error_L[0] / mean_L[0]) ** 2)
        error_.append(f * a * 100)
    return mean_, error_


def data_extracted_from_software(fov_p, param):
    dic_dt_channel = {}
    dic_dt = []

    with open(r'.\files' + '\\' + fov_p + '_' + param + '_per_traces_per_channel_molyso.txt') as inputfile:
        for line in inputfile:
            line_s = line.split('\t')
            channel = int(line_s[0])
            dt = line_s[1].split(',')
            dt.pop()
            dt = list(map(float, dt))
            dic_dt_channel[channel] = dt
            dic_dt = dic_dt + dt
    return dic_dt_channel, dic_dt


def L_from_software(fov_p):
    dic_L_channel = {}
    dic_L = []
    with open(r'.\files' + '\\' + fov_p + '_length_per_traces_per_channel_molyso.txt') as inputfile:
        for line in inputfile:
            line_s = line.split('\t')
            channel = int(line_s[0])
            L = line_s[1].split(',')
            L.pop()
            L = list(map(float, L))
            dic_L_channel[channel] = L
            dic_L = dic_L + L
    return dic_L_channel, dic_L


def elongation_from_software(fov_p):
    dic_L_channel = {}
    dic_L = []
    with open(r'.\files' + '\\' + fov_p + '_elongation_simple_per_channel_molyso.txt') as inputfile:
        for line in inputfile:
            line_s = line.split('\t')
            channel = int(line_s[0])
            L = line_s[1].split(',')
            L.pop()
            L = list(map(float, L))
            dic_L_channel[channel] = L
            dic_L = dic_L + L
    return dic_L_channel, dic_L


def reduced_list(lista, new_size):
    random_index = random.sample(range(0, len(lista)), new_size)
    random_events = [lista[r] for r in random_index]
    return random_events


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def stater_plot(fov_p):
    dt_molyso_mean_pch = []

    for ch in channels[fov_p]:
        dt_molyso_mean_pch.append(numpy.mean(dt_molyso[fov_p][ch]))

    pylab.plot(channels[fov_p], dt_molyso_mean_pch, label='molyso')
    pylab.legend()
    pylab.show()


def print_stats(list_1):
    print("\t N\t", len(list_1))
    print("\t mean\t", pylab.mean(list_1))
    print("\t error\t", pylab.std(list_1) / pylab.sqrt(len(list_1)))


def error_propagation_from_mean(list_error):
    std_m = 0
    for i in range(0, len(list_mean)):
        std_m = std_m + list_mean[i] ** 2
    return numpy.sqrt(std_m) / len(list_error)


def bightness_fov_channel(fov_p):
    dic_b_channel = {}
    dic_b = []
    with open(r'.\files\\' + fov_p + '_brightness_channel.txt') as inputfile:
        for line in inputfile:
            line_s = line.split('\t')
            channel = int(line_s[0])
            b = float(line_s[1])
            dic_b_channel[channel] = b
            dic_b.append(b)
    return dic_b_channel, dic_b


def results_filtered_for_fov(path_file, name_top, sector, fovs, position, channels_to_remove,  cell_params):

    import pandas

    def evaluate_if_mother_in_channel(series):
        min_pos = series.min()
        return list(map(lambda x: x == min_pos, series))


    division_time_low_threshold=cell_params[0]
    large_size_threshold=cell_params[1]
    division_time_max_threshold=cell_params[2]
    threshold_timepoint_discard_empty_channels=cell_params[3]

    # results_raw = pandas.DataFrame()
    # results_filtered = pandas.DataFrame()
    # results_cleaned = pandas.DataFrame()
    results_filt_cleaned = pandas.DataFrame()

    for fov in fovs:
        if fov in channels_to_remove[position].keys():
            ##########################

            file_name = get_name(path_file, fov, name_top)
            results_per_fov = get_results_info(file_name, pandas)
            # results_per_fov = results_per_fov.query("timepoint_num<=349")
            results_per_fov["sector"] = sector
            results_per_fov["fov"] = fov
            results_per_fov["position_on_device"] = position
            results_per_fov["fov_position"] = fov + '_' + position

            set_unique_id_on_device(results_per_fov)
            set_elongation_rate(results_per_fov)
            set_growth_rate(results_per_fov)


            group_channels = results_per_fov.groupby(["channel_in_multipoint", "timepoint_num"])
            results_per_fov["mother"] = group_channels.cellyposition.transform(evaluate_if_mother_in_channel)


            ##########################

            #             results_raw = results_raw.append(results_per_fov)
            #             results_raw = results_raw.append(ma.remove_mistakes_light(result_per_fov))

            results_per_fov_filter = results_per_fov  # .query("timepoint_num<=349")
            results_per_fov_filter = remove_channels(results_per_fov_filter, channels_to_remove[position][fov])
            results_per_fov_filter = remove_ugly_channels(results_per_fov_filter,
                                                             threshold_timepoint_discard_empty_channels, fov)

            # cleaning from
            results_filt_cleaned = results_filt_cleaned.append(
                remove_mistakes(results_per_fov_filter, division_time_low_threshold, large_size_threshold,
                                   division_time_max_threshold))
            # results_filtered = results_filtered.append(ma.remove_mistakes_light(results_per_fov_filter))
            # results_cleaned = results_cleaned.append(ma.remove_mistakes(results_per_fov, division_time_low_threshold, large_size_threshold, division_time_max_threshold))



    def derivative(series):
        if len(series) > 1:
            return numpy.gradient(series)
        else:
            return 0

    def positive_value(series):
        '''
        Evaluation of the sign of the men of the each number of the serie
        :param series:
        :return: True if the mean value is positive, False otherwise
        '''
        mean_serie = numpy.mean(series)
        return mean_serie > 1

    def about_to_divide(series):
        return max(series)

    def newborn(series):
        return min(series)

    def difference(series):
        return max(series) - min(series)


    results_ = results_filt_cleaned
    traces = results_.groupby("unique_id_on_device")

    results_["added_length"] = traces.length.transform(newborn)
    results_["length_newborn"] = traces.length.transform(difference)
    results_["length_about_to_divide"] = traces.length.transform(about_to_divide)

    ##########################
    results_["fluorescence_0"] = results_["fluorescence_raw_0"] - 500
    ##########################


    if 'fluorescence_0' in results_.keys():
        results_["deriv_fls_0"] = traces.fluorescence_raw_0.transform(derivative)
        results_["concentration_fls_0"] = results_["deriv_fls_0"] + results_["fluorescence_raw_0"] * results_["growth_rate"]
        results_["fls_0_ON"] = traces.deriv_fls_0.transform(positive_value)



    if 'fluorescence_1' in results_.keys():
        results_["deriv_fls_1"] = traces.fluorescence_raw_1.transform(derivative)
        results_["concentration_fls_1"] = results_["deriv_fls_1"] + results_["fluorescence_raw_1"] * results_["growth_rate"]
        results_["fls_1_ON"] = traces.deriv_fls_1.transform(positive_value)


    return [results_]


def classify_cells(results_, dic_fls_names):

    def evaluation_strain(b):
        strain_type = 'error'
        for min_b, max_b in dic_fls_names.keys():
            if b >= min_b and b <= max_b:
                strain_type = dic_fls_names[min_b, max_b]
        return strain_type

    def evaluate_series(series):
        return list(map(evaluation_strain, series))


    group_channels_fov = results_.groupby(["channel_in_multipoint", "fov"])
    results_['channel_fluorescence_0'] = group_channels_fov.fluorescence_0.transform('mean')
    results_['cell_type'] = group_channels_fov.channel_fluorescence_0.transform(evaluate_series)
