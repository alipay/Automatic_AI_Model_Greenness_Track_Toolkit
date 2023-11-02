import urllib3

import urllib3.exceptions

# world average co2 intensity
# https://www.iea.org/reports/global-energy-co2-status-report-2019/emissions
# unit gCO2.eq/kWh
CO2_INTENSITY = 475

# https://www.geojs.io/
GEO_JS_URL = "https://get.geojs.io/v1/ip/geo.json"
GEO_JS_COUNTRY_CODE = "country_code"
# https://docs.co2signal.com/#introduction
CO2SIGNAL_API = "https://api.co2signal.com/v1/latest"


class CO2Intensity:
    def __init__(self, co2signal_api_key=None):
        from greenness_track_toolkit.utils import get_logger
        self._co2signal_api_key = co2signal_api_key
        self.value: float = CO2_INTENSITY
        if self._co2signal_api_key:
            try:
                self.value = self.get_co2_signal
            except urllib3.exceptions.HTTPError as error:
                get_logger().error(f"Getting co2 signal from {CO2SIGNAL_API} error:{error}")

    @property
    def get_value(self):
        return self.value

    @property
    def get_computer_location(self):
        from greenness_track_toolkit.utils import pool
        import json
        body = pool.request(method="GET", url=GEO_JS_URL)
        json_data = json.loads(body.data.decode('utf-8'))
        return json_data[GEO_JS_COUNTRY_CODE]

    @property
    def get_co2_signal(self):
        import json
        from greenness_track_toolkit.utils import pool
        location = self.get_computer_location
        url = f"{CO2SIGNAL_API}?countryCode={location}"
        json_data = json.loads(
            pool.request(
                method="GET",
                url=url,
                headers={"auth-token", self._co2signal_api_key}
            ).data.decode(
                'utf-8'
            )
        )
        return json_data['data']['carbonIntensity']