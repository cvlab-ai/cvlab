class ProcessingTimeInfo:

    def __init__(self, start, end, units_count, previous_time_infos):
        milis = int(1000 * (end - start))
        units = units_count if units_count != 0 else 1
        self.work_time = milis
        self.work_time_per_unit = int(float(milis) / units)
        self.total_work_time = milis
        if len(previous_time_infos):
            previous_work_time = 0
            for time in previous_time_infos:
                if time is not None:
                    previous_work_time = max(previous_work_time, time.total_work_time)
                    self.total_work_time += previous_work_time
        self.total_work_time_per_unit = int(round(self.total_work_time / float(units)))

