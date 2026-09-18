_CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "custom_calculator",
        "description": "Perform custom calculations with special formatting",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                },
                "format": {
                    "type": "string",
                    "enum": ["decimal", "scientific", "fraction"],
                    "description": "Output format for the result",
                },
            },
            "required": ["expression"],
        },
    },
}

_BRAVE_WEB_SEARCH_ARGS = '{"query": "latest weather in Tokyo", "country": "JP", "search_lang": "en", "count": 5}'

_TOOL_CALL = {
    "id": "chatcmpl-tool-8063270684d41345",
    "type": "function",
    "function": {
        "name": "brave_web_search",
        "arguments": _BRAVE_WEB_SEARCH_ARGS,
    },
    "index": 0,
}

_ACCUWEATHER_FAVICON = (
    "https://imgs.search.brave.com/fs6uyhM5xA6gctiAKJTHhWtpR2YRWceKfG_9aqjmfRs"
    "/rs:fit:32:32:1:0/g:ce/aHR0cDovL2Zhdmlj/b25zLnNlYXJjaC5i/cmF2ZS5jb20vaWNv"
    "/bnMvNDk4NjU3ZjZm/N2MzYmI3ZjViZjVk/MDcyNDdiNzNlNWM2/MWM3ZTc0MjY3MTY4"
    "/YjNkYWY3ZGQyNzlh/OGFlNmQzZi93d3cu/YWNjdXdlYXRoZXIu/Y29tLw"
)

_TEXT_TOOL_CONTENT = (
    "Found 5 search results: Tokyo, Tokyo, Japan Weather Forecast | AccuWeather,"
    " Japan Meteorological Agency | Weather forecast, Tokyo, Tokyo, Japan Current"
    " Weather | AccuWeather and 2 more\n\n"
    "<search_result citation_number=1>**Tokyo, Tokyo, Japan Weather Forecast | AccuWeather**\n"
    "https://www.accuweather.com/en/jp/tokyo/226396/weather-forecast/226396\n\n"
    "Additional snippets:\n"
    "- Body recovered from frozen Potomac River in Washington, D.C. 1 day ago · Winter Weather"
    " · Snow piles nearly 7 feet high as deadly storms bury northern Japan · 2 days ago"
    " · World Asia Japan Tokyo Tokyo · Funabashi-shi, Chiba · Hachioji-shi, Tokyo"
    " · Kawasaki-shi, Kanagawa ·\n"
    "- Body recovered from frozen Potomac River in Washington, D.C. ... For Business For"
    " Partners For Advertising AccuWeather APIs AccuWeather Connect Personal Weather Stations\n"
    "- Try searching for a city, zip code or point of interest. ... Tokyo, Tokyo Weather Today"
    " WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast® Monthly Air Quality"
    " Health & Activities\n"
    "- Try searching for a city, zip code or point of interest. settings · Tokyo, Tokyo Weather"
    " Today WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast® Monthly Air"
    " Quality Health & Activities · News · For Business · WinterCast Hourly Daily Radar"
    " MinuteCast® Monthly Air Quality Health & Activities ·\n\n"
    "Full page content:\n"
    "Tokyo, Tokyo, Japan Weather Forecast | AccuWeather Go Back For Business | Warnings Data"
    " Suite Forensics Advertising Superior Accuracy™ Tokyo, Tokyo 45° F Location Chevron down"
    " Location News Videos Use Current Location Recent Tokyo Tokyo 45° No results found. Try"
    " searching for a city, zip code or point of interest. Create Your Account Unlock extended"
    " daily and hourly forecasts — all with your free account. Let's Go Chevron right Have an"
    " account already? Log In settings Tokyo, Tokyo Weather Today WinterCast Local {stormName}"
    " Tracker Hourly Daily Radar MinuteCast® Monthly Air Quality Health & Activities Around the"
    " Globe Hurricane Tracker Severe Weather Radar & Maps News News & Features Astronomy"
    " Business Climate Health Recreation Sports Travel For Business Warnings Data Suite"
    " Forensics Advertising Superior Accuracy™ Video Today WinterCast Hourly Daily Radar"
    " MinuteCast® Monthly Air Quality Health & Activities 11 Gale Advisory Tonight's Weather"
    " Fri, Feb 6 Windy with increasing cloudiness Lo: 36° Tomorrow: Breezy in the morning;"
    " cloudy and cooler with a bit of snow with little or no accumulation Hi: 41° Current"
    " Weather 8:05 PM 45° F RealFeel® 33° Clear More Details Wind NW 17 mph Wind Gusts 24 mph"
    " Air Quality Unhealthy Looking Ahead An inch or two of snow Saturday morning through"
    " Saturday evening WinterCast® Sat Morning - Late Sat Night Snow Lasting 18 hours Tokyo"
    " Weather Radar Static Radar Temporarily Unavailable Thank you for your patience as we work"
    " to get everything up and running again. Refresh Page Clouds Temperature Hourly Weather"
    " Chevron left 9 PM 45° 0% 10 PM 42° 0% 11 PM 41° 0% 12 AM 41° 2% 1 AM 40° 5% 2 AM 40°"
    " 5% 3 AM 40° 5% 4 AM 39° 5% 5 AM 39° 5% 6 AM 39° 8% 7 AM 37° 40% 8 AM 36° 41% Chevron"
    " right 10-Day Weather Forecast Tonight 2/6 36° Lo Windy with increasing clouds 25% Sat"
    " 2/7 41° 29° Cooler with a bit of snow Night: Cold; wet snow in the evening 56% Sun 2/8"
    ' 35° 26° Cold with snow, 1-2" Cold; windy in the evening 94% Mon 2/9 48° 31° Not as cold'
    " Clear and chilly 0% Tue 2/10 53° 40° Times of clouds and sun Mostly cloudy 0% Wed 2/11"
    " 55° 38° Rain Partly cloudy 94% Thu 2/12 53° 37° Mostly sunny Cloudy most of the time 2%"
    " Fri 2/13 53° 39° Times of clouds and sun Clearing 1% Sat 2/14 52° 41° Plenty of sunshine"
    " Increasing cloudiness 1% Sun 2/15 54° 42° Low clouds A shower early; mostly cloudy 11%\n"
    "</search_result>\n\n"
    "<search_result citation_number=2>**Japan Meteorological Agency | Weather forecast**\n"
    "https://www.data.jma.go.jp/multi/yoho/yoho_detail.html?code=130010&lang=en\n\n"
    "Full page content:\n"
    "日本語 English 简体中文 繁體中文 한국어 Español Português Bahasa Indonesia Tiếng Việt Tagalog"
    " ภาษาไทย नेपाली भाषा ភាសាខ្មែរ မြန်မာဘာသာ Монгол хэл 00-06 --% 06-12 --% 12-18 --% 18-24"
    " --% 00-06 --% 06-12 --% 12-18 --% 18-24 --% --% --% --% --% --% --% --% - - - - - - - -"
    " - - - - - - topへ\n"
    "</search_result>\n\n"
    "<search_result citation_number=3>**Tokyo, Tokyo, Japan Current Weather | AccuWeather**\n"
    "https://www.accuweather.com/en/jp/tokyo/226396/current-weather/226396\n\n"
    "Additional snippets:\n"
    "- Trash bin lost in Hurricane Sally makes 5-year trek to United Kingdom · 4 days ago"
    " · Winter Weather · Snow piles nearly 7 feet high as deadly storms bury northern Japan"
    " · 1 day ago · World Asia Japan Tokyo Tokyo · Funabashi-shi, Chiba · Hachioji-shi, Tokyo ·\n"
    "- Body recovered from frozen Potomac River in Washington, D.C. ... For Business For"
    " Partners For Advertising AccuWeather APIs AccuWeather Connect Personal Weather Stations"
    " ... AccuWeather Ready Business Health Hurricane Leisure and Recreation Severe Weather"
    " Space and Astronomy Sports Travel Weather News\n"
    "- Try searching for a city, zip code or point of interest. ... Tokyo, Tokyo Weather Today"
    " WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast® Monthly Air Quality"
    " Health & Activities\n"
    "- Try searching for a city, zip code or point of interest. settings · Tokyo, Tokyo Weather"
    " Today WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast® Monthly Air"
    " Quality Health & Activities · News · For Business · Today WinterCast Hourly Daily Radar"
    " MinuteCast® Monthly Air Quality Health & Activities ·\n\n"
    "Full page content:\n"
    "Tokyo, Tokyo, Japan Current Weather | AccuWeather Go Back For Business | Warnings Data"
    " Suite Forensics Advertising Superior Accuracy™ Tokyo, Tokyo 45° F Location Chevron down"
    " Location News Videos Use Current Location Recent Tokyo Tokyo 45° No results found. Try"
    " searching for a city, zip code or point of interest. Create Your Account Unlock extended"
    " daily and hourly forecasts — all with your free account. Let's Go Chevron right Have an"
    " account already? Log In settings Tokyo, Tokyo Weather Today WinterCast Local {stormName}"
    " Tracker Hourly Daily Radar MinuteCast® Monthly Air Quality Health & Activities Around the"
    " Globe Hurricane Tracker Severe Weather Radar & Maps News News & Features Astronomy"
    " Business Climate Health Recreation Sports Travel For Business Warnings Data Suite"
    " Forensics Advertising Superior Accuracy™ Video Today WinterCast Hourly Daily Radar"
    " MinuteCast® Monthly Air Quality Health & Activities Friday, February 6 Current Weather"
    " 8:05 PM 45° F Clear RealFeel® 33° Cold RealFeel Guide Cold 25° to 39° Coats and hats are"
    " appropriate, consider gloves and a scarf. LEARN MORE RealFeel® 33° Wind NW 17 mph Wind"
    " Gusts 24 mph Humidity 47% Indoor Humidity 35% (Dry) Dew Point 26° F Pressure ↑ 29.77 in"
    " Cloud Cover 0% Visibility 15 mi Cloud Ceiling 40000 ft Night 2/6 36° Lo RealFeel® 27°"
    " Cold RealFeel Guide Cold 25° to 39° Coats and hats are appropriate, consider gloves and a"
    " scarf. LEARN MORE Windy with increasing cloudiness alerts Gale Advisory 6:00 PM Friday -"
    " 6:00 AM Saturday alerts Gale Advisory 6:00 PM Friday - 6:00 AM Saturday alerts Gale"
    " Advisory 6:00 PM Friday - 6:00 AM Saturday alerts Gale Advisory 6:00 PM Friday - 6:00 AM"
    " Saturday alerts Gale Advisory 6:00 PM Friday - 6:00 AM Saturday alerts Gale Advisory"
    " 6:00 PM Friday - 6:00 AM Saturday alerts Gale Advisory 6:00 PM Friday - 6:00 AM Saturday"
    " alerts Gale Advisory 6:00 PM Friday - 6:00 AM Saturday alerts Gale Advisory 6:00 PM"
    " Friday - 6:00 AM Saturday alerts High Wave Advisory 6:00 PM Friday - 6:00 AM Saturday"
    " alerts Dry Air Advisory 12:00 AM Friday - 12:00 AM Sunday Wind NNW 17 mph Wind Gusts"
    " 22 mph Probability of Precipitation 25% Probability of Thunderstorms 0% Precipitation"
    " 0.00 in Cloud Cover 76% Evening Overnight Sun & Moon 10 hrs 36 mins Rise 6:37 AM Set"
    " 5:13 PM Waning Gibbous Rise 9:48 PM Set 9:09 AM Temperature History 2/6 High Low Forecast"
    " 57° 36° Average 48° 37° Last Year 49° 31° Further Ahead Hourly Daily Monthly Around the"
    " Globe Hurricane Tracker Severe Weather Radar & Maps News Video Top Stories Winter Weather"
    " Northeast braces for coldest weekend of winter with snow for some 10 minutes ago Winter"
    " Weather Florida growers battle rare freeze, threatening crops 1 day ago Winter Weather"
    " Frigid air eases in second week of February for Midwest, East 1 hour ago Winter Weather"
    " New Jersey firefighter dies after falling into frozen Delaware River 14 hours ago Winter"
    " Weather 'Like a bomb went off:' ESPN's Kirk Herbstreit describes ice storm dam... 2 days"
    " ago More Stories Featured Stories Astronomy 6 planets, moon will align in February, but"
    " there's a catch 20 hours ago Recreation Death Valley seeks tips after illegal off-roading"
    " damages rare plants 20 hours ago Weather News Teen swam hours to get help for family swept"
    " out to sea 1 day ago Winter Weather Body recovered from frozen Potomac River in Washington,"
    " D.C. 1 day ago Winter Weather Snow piles nearly 7 feet high as deadly storms bury northern"
    " Japan 2 days ago World Asia Japan Tokyo Tokyo Weather Near Tokyo: Funabashi-shi , Chiba"
    " Hachioji-shi , Tokyo Kawasaki-shi , Kanagawa Company Proven Superior Accuracy™ About"
    " AccuWeather Digital Advertising Careers Press Contact Us Products & Services For Business"
    " For Partners For Advertising AccuWeather APIs AccuWeather Connect Personal Weather Stations"
    " Apps & Downloads iPhone App Android App See all Apps & Downloads Subscription Services"
    " AccuWeather Premium AccuWeather Professional More AccuWeather Ready Business Health"
    " Hurricane Leisure and Recreation Severe Weather Space and Astronomy Sports Travel Weather"
    " News Company Proven Superior Accuracy™ About AccuWeather Digital Advertising Careers Press"
    " Contact Us Products & Services For Business For Partners For Advertising AccuWeather APIs"
    " AccuWeather Connect Personal Weather Stations Apps & Downloads iPhone App Android App See"
    " all Apps & Downloads Subscription Services AccuWeather Premium AccuWeather Professional"
    " More AccuWeather Ready Business Health Hurricane Leisure and Recreation Severe Weather"
    ' Space and Astronomy Sports Travel Weather News © 2026 AccuWeather, Inc. "AccuWeather" and'
    " sun design are registered trademarks of AccuWeather, Inc. All Rights Reserved. Terms of"
    " Use | Privacy Policy | Cookie Policy | About Your Privacy Do Not Sell or Share My Personal"
    " Information | Data Sources ... ... ... ... ...\n"
    "</search_result>\n\n"
    "<search_result citation_number=4>**Tokyo, Tokyo, Japan Daily Weather | AccuWeather**\n"
    "https://www.accuweather.com/en/jp/tokyo/226396/daily-weather-forecast/226396\n\n"
    "Additional snippets:\n"
    "- Body recovered from frozen Potomac River in Washington, D.C. 1 day ago · Winter Weather"
    " · Snow piles nearly 7 feet high as deadly storms bury northern Japan · 2 days ago"
    " · World Asia Japan Tokyo Tokyo · Funabashi-shi, Chiba · Hachioji-shi, Tokyo"
    " · Kawasaki-shi, Kanagawa ·\n"
    "- Body recovered from frozen Potomac River in Washington, D.C. ... For Business For"
    " Partners For Advertising AccuWeather APIs AccuWeather Connect Personal Weather Stations"
    " ... AccuWeather Ready Business Health Hurricane Leisure and Recreation Severe Weather"
    " Space and Astronomy Sports Travel Weather News\n"
    "- Try searching for a city, zip code or point of interest. ... Tokyo, Tokyo Weather Today"
    " WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast® Monthly Air Quality"
    " Health & Activities\n"
    "- Try searching for a city, zip code or point of interest. settings · Tokyo, Tokyo Weather"
    " Today WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast® Monthly Air"
    " Quality Health & Activities · News · For Business · Today WinterCast Hourly · Radar"
    " MinuteCast® Monthly Air Quality Health & Activities ·\n"
    "</search_result>\n\n"
    "<search_result citation_number=5>**Tokyo, 13, JP Current Weather - The Weather Network**\n"
    "https://www.theweathernetwork.com/en/city/jp/tokyo/tokyo/current\n\n"
    "Additional snippets:\n"
    "- Tokyo, 13, JP · Updated 13 minutes ago · 5°C · Clear · Feels 1 · H: 10° L: 2°"
    " · Full 72 hours · 1am · 3° · Feels 0 · 0% 2am · 3° · Feels -1 · 0% 3am · 3°"
    " · Feels -1 · 10% 4am · 2° · Feels -1 · 20% 5am · 2° · Feels -1 · 20% -- Sunrise ·\n"
    "</search_result>\n\n"
    "Citation Requirements.\n\n"
    "When sourcing statements from references received from Brave Search you MUST provide"
    " citations in accordance with the guidelines below.\n"
    "These instructions only refer to how to provide citations and should not impact the style"
    " of output in any other way.\n"
    "Ensure guidance on providing a rich output is still followed.\n\n"
    "CitationBehaviour:\n"
    "   - CitationUnit:\n"
    "     description: Largest to smallest citable information snippets\n"
    "     options:\n"
    "      - Paragraph\n"
    "      - Sentence\n"
    "      - Clause \n"
    "      - Claim\n\n"
    "    - CitationContent:\n"
    "      description: Allowable information to be cited\n"
    "      options:\n"
    "       - Facts\n"
    "       - Verifiable Information \n"
    "       - Quotes\n\n"
    "   - Rules:\n"
    "     MUST:\n"
    "      - Citations must be in the format [number]\n"
    "      - Citations must ONLY be attached for CitationContent that is within a CitationUnit\n"
    "      - Citations must be at the end of a CitationUnit without spaces\n"
    "      - ALWAYS when multiple consecutive sentences come from the same source, use ONLY ONE"
    " citation at the end of the final sentence from that source\n"
    "      - Attempt to cite the single largest CitationUnit possible with a citation.\n\n"
    "      SHOULD:\n"
    "      - Use multiple citations when information comes from multiple sources\n"
    "      - Reorganize sentences to group same-source information together when possible and"
    " cite once at the end of the group\n\n"
    "      NEVER:\n"
    "      - Citations are never combined like [1, 3] \n"
    "      - Include footnotes or references at the end\n"
    "      - define citations\n"
    "      - overuse citations\n\n"
    "   - Examples:\n"
    "      - Correct:\n"
    "         - Mount Everest is the highest mountain on Earth[1].\n"
    "         - The mountain was first climbed in 1953[1] by Edmund Hillary[2].\n"
    "         - The peak is 8,848 meters high[1][3].\n"
    "         - Mount Everest is the highest peak, it stands at 8,848 meters."
    " The mountain is in the Himalayas[1].\n"
    "         - Mount Everest is the highest peak, it stands at 8,848 meters."
    " Everest is in the Himalayas in Nepal[1]."
    " The mountain was climbed in 1953 by Edmund Hillary[2].\n"
    "      - Incorrect:\n"
    "         - [1] Mount Everest is the highest mountain.\n"
    "         - The peak is 8,848 meters high[1, 3].\n"
    "         - [1] \n"
    "         - Mount Everest is the highest peak[1]. It stands at 8,848 meters[1]."
    " The mountain is in the Himalayas[1].\n"
    "         - Mount Everest is the highest peak, it stands at 8,848 meters[1]."
    " Everest is in the Himalayas in Nepal[1]."
    " The mountain was climbed in 1953 by Edmund Hillary[2]."
)

prompts = [
    {
        "requirements": {"tools": True},
        "prompt": {
            "messages": [
                {
                    "role": "user",
                    "content": "Search for information about quantum computing",
                }
            ],
            "tools": [],
            "brave_mcp_tools_include": ["all"],
        },
    },
    {
        "requirements": {"tools": True},
        "prompt": {
            "messages": [
                {"role": "user", "content": "Search for the latest news in the UK"}
            ],
            "tools": [],
            "brave_mcp_tools_exclude": ["all"],
        },
    },
    {
        "requirements": {"tools": True},
        "prompt": {
            "messages": [
                {
                    "role": "user",
                    "content": "Calculate 123 * 123512 and use your web search tool to search for the latest news",
                }
            ],
            "tools": [_CALCULATOR_TOOL],
            "brave_mcp_tools_include": ["brave_web_search"],
        },
    },
    {
        "requirements": {"tools": True},
        "prompt": {
            "messages": [{"role": "user", "content": "Calculate 4214 * 124214"}],
            "tools": [_CALCULATOR_TOOL],
            "brave_mcp_tools_include": ["brave_web_search"],
        },
    },
    {
        "requirements": {"tools": True},
        "prompt": {
            "messages": [
                {
                    "role": "user",
                    "content": "Use your web search tool to search for the latest news",
                }
            ],
            "tools": [_CALCULATOR_TOOL],
            "brave_mcp_tools_include": ["brave_web_search"],
        },
    },
    {
        "requirements": {"tools": True},
        "prompt": {
            "messages": [
                {
                    "role": "user",
                    "content": "Search for recent news in the UK and use web search to get weather information",
                }
            ],
            "tools": [],
            "brave_mcp_tools_include": ["brave_news_search"],
        },
    },
    {
        "requirements": {"tools": True},
        "prompt": {
            "messages": [
                {"role": "user", "content": "Search for latest weather in Tokyo"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_TOOL_CALL],
                },
                {
                    "role": "tool",
                    "tool_call_id": "chatcmpl-tool-8063270684d41345",
                    "content": [
                        {
                            "type": "brave-chat.webSources",
                            "sources": [
                                {
                                    "title": "Tokyo, Tokyo, Japan Weather Forecast | AccuWeather",
                                    "url": "https://www.accuweather.com/en/jp/tokyo/226396/weather-forecast/226396",
                                    "favicon": _ACCUWEATHER_FAVICON,
                                    "page_content": (
                                        "Tokyo, Tokyo, Japan Weather Forecast | AccuWeather Go Back For Business"
                                        " | Warnings Data Suite Forensics Advertising Superior Accuracy™ Tokyo,"
                                        " Tokyo 45° F Location Chevron down Location News Videos Use Current"
                                        " Location Recent Tokyo Tokyo 45° No results found. Try searching for a"
                                        " city, zip code or point of interest. Create Your Account Unlock extended"
                                        " daily and hourly forecasts — all with your free account. Let's Go Chevron"
                                        " right Have an account already? Log In settings Tokyo, Tokyo Weather Today"
                                        " WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast®"
                                        " Monthly Air Quality Health & Activities Around the Globe Hurricane Tracker"
                                        " Severe Weather Radar & Maps News News & Features Astronomy Business"
                                        " Climate Health Recreation Sports Travel For Business Warnings Data Suite"
                                        " Forensics Advertising Superior Accuracy™ Video Today WinterCast Hourly"
                                        " Daily Radar MinuteCast® Monthly Air Quality Health & Activities 11 Gale"
                                        " Advisory Tonight's Weather Fri, Feb 6 Windy with increasing cloudiness"
                                        " Lo: 36° Tomorrow: Breezy in the morning; cloudy and cooler with a bit of"
                                        " snow with little or no accumulation Hi: 41° Current Weather 8:05 PM 45° F"
                                        " RealFeel® 33° Clear More Details Wind NW 17 mph Wind Gusts 24 mph Air"
                                        " Quality Unhealthy Looking Ahead An inch or two of snow Saturday morning"
                                        " through Saturday evening WinterCast® Sat Morning - Late Sat Night Snow"
                                        " Lasting 18 hours Tokyo Weather Radar Static Radar Temporarily Unavailable"
                                        " Thank you for your patience as we work to get everything up and running"
                                        " again. Refresh Page Clouds Temperature Hourly Weather Chevron left 9 PM"
                                        " 45° 0% 10 PM 42° 0% 11 PM 41° 0% 12 AM 41° 2% 1 AM 40° 5% 2 AM 40° 5%"
                                        " 3 AM 40° 5% 4 AM 39° 5% 5 AM 39° 5% 6 AM 39° 8% 7 AM 37° 40% 8 AM 36°"
                                        " 41% Chevron right 10-Day Weather Forecast Tonight 2/6 36° Lo Windy with"
                                        " increasing clouds 25% Sat 2/7 41° 29° Cooler with a bit of snow Night:"
                                        ' Cold; wet snow in the evening 56% Sun 2/8 35° 26° Cold with snow, 1-2"'
                                        " Cold; windy in the evening 94% Mon 2/9 48° 31° Not as cold Clear and"
                                        " chilly 0% Tue 2/10 53° 40° Times of clouds and sun Mostly cloudy 0% Wed"
                                        " 2/11 55° 38° Rain Partly cloudy 94% Thu 2/12 53° 37° Mostly sunny Cloudy"
                                        " most of the time 2% Fri 2/13 53° 39° Times of clouds and sun Clearing 1%"
                                        " Sat 2/14 52° 41° Plenty of sunshine Increasing cloudiness 1% Sun 2/15"
                                        " 54° 42° Low clouds A shower early; mostly cloudy 11%"
                                    ),
                                    "extra_snippets": [
                                        "Body recovered from frozen Potomac River in Washington, D.C. 1 day ago"
                                        " · Winter Weather · Snow piles nearly 7 feet high as deadly storms bury"
                                        " northern Japan · 2 days ago · World Asia Japan Tokyo Tokyo"
                                        " · Funabashi-shi, Chiba · Hachioji-shi, Tokyo · Kawasaki-shi, Kanagawa ·",
                                        "Body recovered from frozen Potomac River in Washington, D.C. ... For"
                                        " Business For Partners For Advertising AccuWeather APIs AccuWeather"
                                        " Connect Personal Weather Stations",
                                        "Try searching for a city, zip code or point of interest. ... Tokyo, Tokyo"
                                        " Weather Today WinterCast Local {stormName} Tracker Hourly Daily Radar"
                                        " MinuteCast® Monthly Air Quality Health & Activities",
                                        "Try searching for a city, zip code or point of interest. settings · Tokyo,"
                                        " Tokyo Weather Today WinterCast Local {stormName} Tracker Hourly Daily"
                                        " Radar MinuteCast® Monthly Air Quality Health & Activities · News · For"
                                        " Business · WinterCast Hourly Daily Radar MinuteCast® Monthly Air Quality"
                                        " Health & Activities ·",
                                    ],
                                },
                                {
                                    "title": "Japan Meteorological Agency | Weather forecast",
                                    "url": "https://www.data.jma.go.jp/multi/yoho/yoho_detail.html?code=130010&lang=en",
                                    "favicon": (
                                        "https://imgs.search.brave.com/3DV2AlVSETIlJxqgbHZ81P3nxxgMFCS1gidR3gg-CW4"
                                        "/rs:fit:32:32:1:0/g:ce/aHR0cDovL2Zhdmlj/b25zLnNlYXJjaC5i/cmF2ZS5jb20vaWNv"
                                        "/bnMvNzMwNThjYTli/YTc0MGFmNjVlZWRl/ZTNkZWI1ZjU2MDQ4/MzlkOWRlZGYxZjAz"
                                        "/Zjc2ZmM1ZjMzMDI1/Mzk0YWVjMy93d3cu/ZGF0YS5qbWEuZ28u/anAv"
                                    ),
                                    "page_content": (
                                        "日本語 English 简体中文 繁體中文 한국어 Español Português Bahasa Indonesia"
                                        " Tiếng Việt Tagalog ภาษาไทย नेपाली भाषा ភាសាខ្មែរ မြန်မာဘာသာ Монгол хэл"
                                        " 00-06 --% 06-12 --% 12-18 --% 18-24 --% 00-06 --% 06-12 --% 12-18 --%"
                                        " 18-24 --% --% --% --% --% --% --% --% - - - - - - - - - - - - - - topへ"
                                    ),
                                    "extra_snippets": None,
                                },
                                {
                                    "title": "Tokyo, Tokyo, Japan Current Weather | AccuWeather",
                                    "url": "https://www.accuweather.com/en/jp/tokyo/226396/current-weather/226396",
                                    "favicon": _ACCUWEATHER_FAVICON,
                                    "page_content": (
                                        "Tokyo, Tokyo, Japan Current Weather | AccuWeather Go Back For Business"
                                        " | Warnings Data Suite Forensics Advertising Superior Accuracy™ Tokyo,"
                                        " Tokyo 45° F Location Chevron down Location News Videos Use Current"
                                        " Location Recent Tokyo Tokyo 45° No results found. Try searching for a"
                                        " city, zip code or point of interest. Create Your Account Unlock extended"
                                        " daily and hourly forecasts — all with your free account. Let's Go Chevron"
                                        " right Have an account already? Log In settings Tokyo, Tokyo Weather Today"
                                        " WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast®"
                                        " Monthly Air Quality Health & Activities Around the Globe Hurricane Tracker"
                                        " Severe Weather Radar & Maps News News & Features Astronomy Business"
                                        " Climate Health Recreation Sports Travel For Business Warnings Data Suite"
                                        " Forensics Advertising Superior Accuracy™ Video Today WinterCast Hourly"
                                        " Daily Radar MinuteCast® Monthly Air Quality Health & Activities Friday,"
                                        " February 6 Current Weather 8:05 PM 45° F Clear RealFeel® 33° Cold"
                                        " RealFeel Guide Cold 25° to 39° Coats and hats are appropriate, consider"
                                        " gloves and a scarf. LEARN MORE RealFeel® 33° Wind NW 17 mph Wind Gusts"
                                        " 24 mph Humidity 47% Indoor Humidity 35% (Dry) Dew Point 26° F Pressure ↑"
                                        " 29.77 in Cloud Cover 0% Visibility 15 mi Cloud Ceiling 40000 ft Night 2/6"
                                        " 36° Lo RealFeel® 27° Cold RealFeel Guide Cold 25° to 39° Coats and hats"
                                        " are appropriate, consider gloves and a scarf. LEARN MORE Windy with"
                                        " increasing cloudiness alerts Gale Advisory 6:00 PM Friday - 6:00 AM"
                                        " Saturday alerts Gale Advisory 6:00 PM Friday - 6:00 AM Saturday alerts"
                                        " Gale Advisory 6:00 PM Friday - 6:00 AM Saturday alerts Gale Advisory"
                                        " 6:00 PM Friday - 6:00 AM Saturday alerts Gale Advisory 6:00 PM Friday -"
                                        " 6:00 AM Saturday alerts Gale Advisory 6:00 PM Friday - 6:00 AM Saturday"
                                        " alerts Gale Advisory 6:00 PM Friday - 6:00 AM Saturday alerts Gale"
                                        " Advisory 6:00 PM Friday - 6:00 AM Saturday alerts Gale Advisory 6:00 PM"
                                        " Friday - 6:00 AM Saturday alerts High Wave Advisory 6:00 PM Friday -"
                                        " 6:00 AM Saturday alerts Dry Air Advisory 12:00 AM Friday - 12:00 AM"
                                        " Sunday Wind NNW 17 mph Wind Gusts 22 mph Probability of Precipitation"
                                        " 25% Probability of Thunderstorms 0% Precipitation 0.00 in Cloud Cover"
                                        " 76% Evening Overnight Sun & Moon 10 hrs 36 mins Rise 6:37 AM Set 5:13 PM"
                                        " Waning Gibbous Rise 9:48 PM Set 9:09 AM Temperature History 2/6 High Low"
                                        " Forecast 57° 36° Average 48° 37° Last Year 49° 31° Further Ahead Hourly"
                                        " Daily Monthly Around the Globe Hurricane Tracker Severe Weather Radar &"
                                        " Maps News Video Top Stories Winter Weather Northeast braces for coldest"
                                        " weekend of winter with snow for some 10 minutes ago Winter Weather Florida"
                                        " growers battle rare freeze, threatening crops 1 day ago Winter Weather"
                                        " Frigid air eases in second week of February for Midwest, East 1 hour ago"
                                        " Winter Weather New Jersey firefighter dies after falling into frozen"
                                        " Delaware River 14 hours ago Winter Weather 'Like a bomb went off:'"
                                        " ESPN's Kirk Herbstreit describes ice storm dam... 2 days ago More Stories"
                                        " Featured Stories Astronomy 6 planets, moon will align in February, but"
                                        " there's a catch 20 hours ago Recreation Death Valley seeks tips after"
                                        " illegal off-roading damages rare plants 20 hours ago Weather News Teen"
                                        " swam hours to get help for family swept out to sea 1 day ago Winter"
                                        " Weather Body recovered from frozen Potomac River in Washington, D.C. 1"
                                        " day ago Winter Weather Snow piles nearly 7 feet high as deadly storms"
                                        " bury northern Japan 2 days ago World Asia Japan Tokyo Tokyo Weather Near"
                                        " Tokyo: Funabashi-shi , Chiba Hachioji-shi , Tokyo Kawasaki-shi , Kanagawa"
                                        " Company Proven Superior Accuracy™ About AccuWeather Digital Advertising"
                                        " Careers Press Contact Us Products & Services For Business For Partners"
                                        " For Advertising AccuWeather APIs AccuWeather Connect Personal Weather"
                                        " Stations Apps & Downloads iPhone App Android App See all Apps & Downloads"
                                        " Subscription Services AccuWeather Premium AccuWeather Professional More"
                                        " AccuWeather Ready Business Health Hurricane Leisure and Recreation Severe"
                                        " Weather Space and Astronomy Sports Travel Weather News Company Proven"
                                        " Superior Accuracy™ About AccuWeather Digital Advertising Careers Press"
                                        " Contact Us Products & Services For Business For Partners For Advertising"
                                        " AccuWeather APIs AccuWeather Connect Personal Weather Stations Apps &"
                                        " Downloads iPhone App Android App See all Apps & Downloads Subscription"
                                        " Services AccuWeather Premium AccuWeather Professional More AccuWeather"
                                        " Ready Business Health Hurricane Leisure and Recreation Severe Weather"
                                        " Space and Astronomy Sports Travel Weather News © 2026 AccuWeather, Inc."
                                        ' "AccuWeather" and sun design are registered trademarks of AccuWeather,'
                                        " Inc. All Rights Reserved. Terms of Use | Privacy Policy | Cookie Policy"
                                        " | About Your Privacy Do Not Sell or Share My Personal Information"
                                        " | Data Sources ... ... ... ... ..."
                                    ),
                                    "extra_snippets": [
                                        "Trash bin lost in Hurricane Sally makes 5-year trek to United Kingdom"
                                        " · 4 days ago · Winter Weather · Snow piles nearly 7 feet high as deadly"
                                        " storms bury northern Japan · 1 day ago · World Asia Japan Tokyo Tokyo"
                                        " · Funabashi-shi, Chiba · Hachioji-shi, Tokyo ·",
                                        "Body recovered from frozen Potomac River in Washington, D.C. ... For"
                                        " Business For Partners For Advertising AccuWeather APIs AccuWeather"
                                        " Connect Personal Weather Stations ... AccuWeather Ready Business Health"
                                        " Hurricane Leisure and Recreation Severe Weather Space and Astronomy"
                                        " Sports Travel Weather News",
                                        "Try searching for a city, zip code or point of interest. ... Tokyo, Tokyo"
                                        " Weather Today WinterCast Local {stormName} Tracker Hourly Daily Radar"
                                        " MinuteCast® Monthly Air Quality Health & Activities",
                                        "Try searching for a city, zip code or point of interest. settings · Tokyo,"
                                        " Tokyo Weather Today WinterCast Local {stormName} Tracker Hourly Daily"
                                        " Radar MinuteCast® Monthly Air Quality Health & Activities · News · For"
                                        " Business · Today WinterCast Hourly Daily Radar MinuteCast® Monthly Air"
                                        " Quality Health & Activities ·",
                                    ],
                                },
                                {
                                    "title": "Tokyo, Tokyo, Japan Daily Weather | AccuWeather",
                                    "url": "https://www.accuweather.com/en/jp/tokyo/226396/daily-weather-forecast/226396",
                                    "favicon": _ACCUWEATHER_FAVICON,
                                    "page_content": None,
                                    "extra_snippets": [
                                        "Body recovered from frozen Potomac River in Washington, D.C. 1 day ago"
                                        " · Winter Weather · Snow piles nearly 7 feet high as deadly storms bury"
                                        " northern Japan · 2 days ago · World Asia Japan Tokyo Tokyo"
                                        " · Funabashi-shi, Chiba · Hachioji-shi, Tokyo · Kawasaki-shi, Kanagawa ·",
                                        "Body recovered from frozen Potomac River in Washington, D.C. ... For"
                                        " Business For Partners For Advertising AccuWeather APIs AccuWeather"
                                        " Connect Personal Weather Stations ... AccuWeather Ready Business Health"
                                        " Hurricane Leisure and Recreation Severe Weather Space and Astronomy"
                                        " Sports Travel Weather News",
                                        "Try searching for a city, zip code or point of interest. ... Tokyo, Tokyo"
                                        " Weather Today WinterCast Local {stormName} Tracker Hourly Daily Radar"
                                        " MinuteCast® Monthly Air Quality Health & Activities",
                                        "Try searching for a city, zip code or point of interest. settings · Tokyo,"
                                        " Tokyo Weather Today WinterCast Local {stormName} Tracker Hourly Daily"
                                        " Radar MinuteCast® Monthly Air Quality Health & Activities · News · For"
                                        " Business · Today WinterCast Hourly · Radar MinuteCast® Monthly Air"
                                        " Quality Health & Activities ·",
                                    ],
                                },
                                {
                                    "title": "Tokyo, 13, JP Current Weather - The Weather Network",
                                    "url": "https://www.theweathernetwork.com/en/city/jp/tokyo/tokyo/current",
                                    "favicon": (
                                        "https://imgs.search.brave.com/PngeKo7zbzNt-2zCSn559wzst7VixpZEDNgaxIMzo-E"
                                        "/rs:fit:32:32:1:0/g:ce/aHR0cDovL2Zhdmlj/b25zLnNlYXJjaC5i/cmF2ZS5jb20vaWNv"
                                        "/bnMvMDcxODFlNWZi/YzQ1OTRhZDJhMGQz/ODQ2MWJkNDY2OTkz/NzYzNTBjMjg0ZTgx"
                                        "/YTc0MDdlNmFhNTNk/ODI0ZTI5OC93d3cu/dGhld2VhdGhlcm5l/dHdvcmsuY29tLw"
                                    ),
                                    "page_content": None,
                                    "extra_snippets": [
                                        "Tokyo, 13, JP · Updated 13 minutes ago · 5°C · Clear · Feels 1"
                                        " · H: 10° L: 2° · Full 72 hours · 1am · 3° · Feels 0 · 0% 2am · 3°"
                                        " · Feels -1 · 0% 3am · 3° · Feels -1 · 10% 4am · 2° · Feels -1 · 20%"
                                        " 5am · 2° · Feels -1 · 20% -- Sunrise ·",
                                    ],
                                },
                            ],
                            "query": "latest weather in Tokyo",
                            "rich_results": [
                                {
                                    "type": "rich",
                                    "subtype": "weather",
                                    "provider": {
                                        "type": "external",
                                        "name": "OpenWeatherMap",
                                        "url": "https://openweathermap.org/",
                                    },
                                    "stopwatch": False,
                                    "weather": {
                                        "location": {
                                            "id": 1850144,
                                            "name": "Tokyo",
                                            "country": "JP",
                                            "state": "",
                                            "coords": {"lat": 35.6897, "lon": 139.6895},
                                            "population": 12445327,
                                            "sunrise": 1770327464,
                                            "sunset": 1770365571,
                                            "tzoffset": 32400,
                                            "implicit_location": False,
                                        },
                                        "current_time_iso": "2026-02-06T20:05:30",
                                        "current_weather": {
                                            "ts": 1770375270,
                                            "sunrise": 1770327464,
                                            "sunset": 1770365570,
                                            "temp": 10.13,
                                            "feels_like": 8.3,
                                            "pressure": 1010,
                                            "humidity": 42,
                                            "dew_point": -1.94,
                                            "uvi": 0,
                                            "clouds": 0,
                                            "visibility": 10000,
                                            "wind": {"speed": 7.2, "deg": 340},
                                            "weather": {
                                                "id": 800,
                                                "main": "clear",
                                                "description": "clear sky",
                                                "icon": "01n",
                                            },
                                        },
                                    },
                                }
                            ],
                        }
                    ],
                },
                {"role": "user", "content": "How high was the snow in northern Japan?"},
            ],
            "tools": [],
            "brave_mcp_tools_include": ["all"],
        },
    },
    {
        "requirements": {"tools": True},
        "prompt": {
            "messages": [
                {"role": "user", "content": "Search for latest weather in Tokyo"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_TOOL_CALL],
                },
                {
                    "role": "tool",
                    "tool_call_id": "chatcmpl-tool-8063270684d41345",
                    "content": _TEXT_TOOL_CONTENT,
                },
                {"role": "user", "content": "How high was the snow in northern Japan?"},
            ],
            "tools": [],
            "brave_mcp_tools_include": ["all"],
        },
    },
]
