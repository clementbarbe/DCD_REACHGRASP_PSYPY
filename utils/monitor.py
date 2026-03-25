from psychopy import monitors

def create_default_monitor():
    mon = monitors.Monitor('temp_monitor')
    mon.setSizePix((3840, 2160))
    mon.setWidth(53)
    mon.setDistance(60)
    mon.saveMon()
    return mon