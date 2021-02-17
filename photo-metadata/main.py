import exiftool
import re
import os

if __name__ == '__main__':

    # files = ["/home/jm/Pictures/Photos/mom-me.jpg"]
    files = os.listdir('/home/jm/Pictures/Photos')

    def addPath(file_name):
        return '/home/jm/Pictures/Photos/' + file_name

    files = map(addPath, files)

    print()
    # for file in files:
    #     print(file)

    times = []

    print(list(files))

    et = exiftool.ExifTool()

    et.start()
    print(et.running)

    print('help')

    for file in files:
        print('help')
        print(file)
        exif_data = et.get_metadata(file)
        data = str(exif_data).split(",")
        print(data)
        for d in data:
            if re.match(".*ProfileDateTime.*", d):
                print(d)
                # time = d

    print(list(times))
    # metadata = et.get_metadata_batch(files)
    # print(metadata.File.FilePermissions)
    # count = list.count(metadata)




