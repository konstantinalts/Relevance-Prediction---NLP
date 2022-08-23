from nltk.stem.porter import *


class DataProcessor():
    def __init__(self):
        self.stemmer = PorterStemmer()

    def str_cleanup(self, s):

        # Remove empty space between Pattern 1 (any number) and Pattern 2 ('.'' followed by numbers)
        # Example: 'Y33 .043' --> 'Y33.043'
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        # Deal with units
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = re.sub(r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1 deg. ", s)
        s = re.sub(r"([0-9]+)( *)(v|volts|volt)\.?", r"\1 volt. ", s)
        s = re.sub(r"([0-9]+)( *)(wattage|watts|watt)\.?", r"\1 watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amp. ", s)
        s = re.sub(r"([0-9]+)( *)(qquart|quart)\.?", r"\1 qt. ", s)
        s = re.sub(r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1 hr ", s)
        s = re.sub(
            r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r"\1 gal. per min. ", s)
        s = re.sub(
            r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1 gal. per hr ", s)
        # Deal with special characters and HTMl syntax
        s = s.replace("$", " ")
        s = s.replace("?", " ")
        s = s.replace("&nbsp;", " ")
        s = s.replace("&amp;", "&")
        s = s.replace("&#39;", "'")
        s = s.replace("/>/Agt/>", "")
        s = s.replace("</a<gt/", "")
        s = s.replace("gt/>", "")
        s = s.replace("/>", "")
        s = s.replace("<br", "")
        s = s.replace("<.+?>", "")
        s = s.replace("[ &<>)(_,;:!?\+^~@#\$]+", " ")
        s = s.replace("'s\\b", "")
        s = s.replace("[']+", "")
        s = s.replace("[\"]+", "")
        s = s.replace("-", " ")
        s = s.replace("+", " ")
        # Remove text between paranthesis/brackets)
        s = s.replace("[ ]?[[(].+?[])]", "")
        # remove sizes
        s = s.replace("size: .+$", "")
        s = s.replace("size [0-9]+[.]?[0-9]+\\b", "")

        return s

    def str_stem(self, s):

        return " ".join([self.stemmer.stem(re.sub('[^A-Za-z0-9-./]', ' ', word)) for word in s.lower().split()])

    def str_process(self, s):

        if isinstance(s, str):
            s = self.str_cleanup(s)
            s = self.str_stem(s)
            return s

        else:
            return "null"

    def df_process(self, df):
        return df.apply(self.str_process())
