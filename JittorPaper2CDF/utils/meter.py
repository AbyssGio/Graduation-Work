

class Statistics:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.count += n
        self.total += (value * n)

    @property
    def global_avg(self):
        if self.count == 0:
            return 0
        else:
            return self.total / self.count


class AggregationMeter:
    def __init__(self, meter_num):
        self.meter = []
        for n in range(meter_num):
            self.meter.append(Statistics())

    def reset(self):
        for s in self.meter:
            s.reset()

    def update(self, n_list):
        for idx, s in enumerate(self.meter):
            s.update(n_list[idx][0], n_list[idx][1])

    @property
    def global_avg(self):
        return [s.global_avg for s in self.meter]

