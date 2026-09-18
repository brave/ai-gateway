import json

import pytest

from aichat.protocol.open_ai_protocol import TextContentPart, UserMessage
from aichat.serve.services.conversation_title import (
    last_message_includes_conversation_title,
)
from aichat.serve.utils import (
    generate_static_content_generator,
    parse_last_user_input,
    static_content_generator,
)


def generate_v2_complete_headers(payload, **_kwargs):
    # Auth verdict is mocked in conftest, so no real signing/digest headers needed.
    return {}


def parseSSE(line):
    return json.loads(line[len("data: ") :])


def test_parse_last_user_input_new_tab_page():
    text = """
[INST] <<SYS>>
The current date is Friday, September 8, 2023.

Your name is Leo, a helpful, respectful and honest AI assistant created by the company Brave. You will be replying to a user of the Brave browser. Always respond in a neutral tone. Keep your answers short and to the point, unless otherwise specified by the user.

Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

sup? [/INST] Here is your response:
"""
    assert parse_last_user_input(text) == "sup?"


def test_parse_last_user_input_article_suggested_questions():
    text = """
[INST] <<SYS>>
Your goal is to answer the user's requests exactly in concise manner.
<</SYS>>

This is an article:

<article>
Hurricane Lee  weakened slightly into a Category 4 storm Friday morning, following a day in which the hurricane strengthened at a historic pace into a powerful Category 5 rarely seen in the Atlantic Ocean. The hurricane is packing destructive maximum sustained winds of 155 mph and is about 550 miles east of the northern Leeward Islands. “Although Lee’s current intensity is lower than the overnight peak, the hurricane remains very powerful,” the National Hurricane Center said, noting that more slight fluctuations in maximum winds are expected over the next several days. Lee is expected to remain a major hurricane over the southwestern Atlantic early next week, though it’s too soon to know whether this system will directly impact the US mainland. Lee , which was a Category 1 storm Thursday, intensified with exceptional speed in warm ocean waters, doubling its wind speeds in just a day. The storm’s winds increased by 85 mph in a 24-hour period, which tied it with Hurricane Matthew for the third-fastest  rapid intensification  in the Atlantic, according to NOAA research meteorologist John Kaplan. The monstrous hurricane  struck Haiti in 2016 , killing hundreds in the Caribbean nation while also wreaking havoc on parts of the US Southeast. Dangerous surf and rip currents will spread across the northern Caribbean on Friday and begin affecting the mainland US on Sunday. The center of Lee will pass to the north of the Leeward Islands, the Virgin Islands and Puerto Rico this weekend and into early next week. Tropical storm conditions, life-threatening surf and rip currents could occur on some of these islands over the weekend. Lee now in rare company Lee hit a rare strength that few storms have ever achieved. Only 2% of storms in the Atlantic reach Category 5 strength, according to NOAA’s hurricane database. Including Lee, only 40 Category 5 hurricanes have roamed the Atlantic since 1924. Category 5 is the highest level on the hurricane wind speed scale and has   no maximum point. Hurricanes hit this level when their sustained winds reach   157 mph or higher. A 165-mph storm like Lee is the same category as Hurricane Allen, the Atlantic’s strongest hurricane on record, which topped out at 190 mph in 1980. Hurricanes need the perfect mixture of warm water, moist air and light upper-level winds to intensify enough to reach Category 5 strength. Lee had all of these, especially warm water amid the warmest summer on record. Sea-surface temperatures across the portion of the Atlantic Ocean that Lee is tracking through are a staggering 2 degrees Celsius (3.6 degrees Fahrenheit) above normal after rising to “far above record levels” this summer, according to David Zierden, Florida’s state climatologist. Reaching Category 5 strength has become more common over the last decade. Lee is the 8th Category 5 since 2016, meaning 20% of these exceptionally powerful hurricanes on record in NOAA’s hurricane database have come in the last seven years. The Atlantic is not the only ocean to have spawned a monster storm in 2023. All seven ocean basins where tropical cyclones can form have had a storm reach Category 5 strength so far this year, including Hurricane Jova, which reached Category 5 status in the eastern Pacific earlier this week. How close will Hurricane Lee get to the US? Computer model trends for Lee have shown the hurricane taking a turn to the north early next week. But exactly when that turn occurs and how far west Lee will manage to track by then will play a huge role in how close it gets to the US. Several steering factors at the surface and upper levels of the atmosphere will determine how close Lee will get to the East Coast. An area of high pressure over the Atlantic, known as the Bermuda High, will have a major influence in how quickly Lee turns. The Bermuda High is expected to remain very strong into the weekend, which will keep Lee on its current west-northwestward track and slow it down a bit. As the high pressure weakens next week it will allow Lee to start moving northward. Once that turn to the north occurs, the position of the jet stream – strong upper-level winds that can change the direction of a hurricane’s path – will influence how closely Lee is steered to the US. Scenario:  Out to Sea Lee could make a quick turn to the north early next week if high pressure weakens significantly. If the jet stream sets up along the East Coast, it will act as a barrier that prevents Lee from approaching the coast. This scenario would keep Lee farther away from the US coast but could bring the storm closer to Bermuda. Scenario:  Close to East Coast Lee could make a slower turn to the north because the high pressure remains robust, and the jet stream sets up farther inland over the Eastern US. This scenario would leave portions of the East Coast, mainly north of the Carolinas, vulnerable to a much closer approach from Lee. All these factors have yet to come into focus, and the hurricane is still at least seven days from being a threat to the East Coast. Any potential US impact will become more clear as the Lee moves west in the coming days. CNN Meteorologist Robert Shackelford and Aya Elamroussi contributed to this report.
</article>

Propose up to 3 very short questions, around 10 words, that a reader may ask about the this article. Consider intriguing or unusual elements of the content, or structurally important. [/INST] Sure, here are three intriguing or unusual questions that a reader may ask about the article: <ul> <li>
"""
    assert parse_last_user_input(text) == ""


def test_parse_last_user_input_article_first_message_2():
    text = """
[INST] <<SYS>>
The current date is Thursday, November 9, 2023.

Your name is Leo, a helpful, respectful and honest AI assistant created by the company Brave. You will be replying to a user of the Brave browser. Always respond in a neutral tone. Be polite and courteous. Answer concisely in no more than 50-80 words.

Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

This is an article within <article> tags:

<article> Live Israel-Hamas war  Live Republican presidential debate Israel-Hamas war rages as outcry grows over Gaza crisis By  Kathleen Magramo ,  Heather Chen , Nadeen Ebrahim, Ed Upright,  Alisha Ebrahimji  and  Adrienne Vogt , CNN What we're covering Residents look for family members in rubble of destroyed buildings in central Gaza From CNN’s Kareem El Damanhoury   Gaza residents are looking for family members within the rubble of destroyed buildings in the central city of Deir al-Balah after what witnesses said was an airstrike on the area, agency videos on AFP showed. Injured people, including  children , were taken to hospitals after a residential building collapsed, video showed. The Hamas-run Interior Ministry in Gaza said people were killed in a strike on Deir al-Balah. It’s unclear if the ministry was citing the AFP incident and CNN could not verify the claim. On Wednesday, the ministry said Israel struck the  Jabalya Refugee Camp  in northern Gaza. CNN has reached out to the Israeli military for comment on the Jabalya strike. In previous statements, the Israeli military has maintained that it targets Hamas infrastructure in the strip.  Israeli and US intelligence heads meet Qatari officials in Doha for hostage negotiations, source says From CNN's Becky Anderson A trilateral meeting with Qatari officials and the intelligence chiefs of Israel and the US was held in Doha on Thursday to discuss hostage releases in exchange for a humanitarian pause and aid entry to Gaza, a diplomatic source familiar with the talks told CNN.  The meeting — which included CIA Director William Burns, Mossad head David Barnea and Qatari officials — discussed a proposed plan to release between 10 to 20 civilian hostages in return for a three-day pause in fighting and the entry of further aid, plus enabling Hamas to compile and hand over a list of hostages being held in Gaza, the source said. A US official confirmed that Burns took part in the meeting with Barnea and the Qatari prime minister concerning hostage issues. The official declined to comment on the terms of what was discussed. On Wednesday, CNN reported that there was no prospect of Israel agreeing to a sustained pause in fighting  without a substantial number of hostages being released , according to one senior US official. The multi-party talks – in which Qatar is playing a key mediating role — have been ongoing for weeks.  CNN previously reported that one Israeli official said that the country was “ready for a pause” if there could be certainty that Hamas was “serious about releasing hostages.” What is not clear is how long of a pause Israel would be willing to agree to and what would amount to an acceptable number of hostages released.  Negotiations have also centered around exchanging hostages for Palestinian prisoners held by Israel, CNN has previously reported. CNN's Alex Marquardt and Katie Bo Lillis contributed reporting to this post. An "unprecedented" amount of reports of anti-Arab and Islamophobic bias in the last month, new data shows From CNN's Chelsea Bailey The Council on American-Islamic Relations (CAIR) has   documented an “appalling” rise in reported anti-Arab and anti-Muslim bias incidents in the month since violence escalated between Israel and Hamas, the organization announced Thursday. The nation’s largest Muslim advocacy group said it has received 1,283 requests for help and reports of bias in the month since the  October 7 Hamas attack on Israel . The organization said in 2022 it received an average of 406 complaints in a 29-day period. The new data, CAIR said, reflects a 216% increase in requests for help and  reported bias incidents  compared to the previous year. Corey Saylor, director of research and advocacy at CAIR, said in a statement shared with CNN that the data represents the largest wave of  Islamophobic  and anti-Arab bias the organization has recorded since then-candidate Donald Trump called for a Muslim Ban in 2015. Throughout the year, CAIR records and monitors incidents of reported anti-Muslim and anti-Arab bias from local chapters across the country. The new data reflects a sharp pivot from CAIR’s cautiously optimistic outlook earlier this year, when the  organization published  a report noting 2022 was the first time the US charted a decrease in anti-Muslim bias incidents since they began tracking such reports in the 1990s. Read more about the  CAIR report  here. Most Gaza hospitals have stopped working, Palestinian officials say. Here's the latest on the Israel-Hamas war From CNN Staff The majority of Gazan hospitals – 18 out of 35 in the Gaza Strip – have now stopped functioning, according to the Palestinian Ministry of Health in Ramallah, which draws figures from the Hamas-controlled territory. As the humanitarian crisis in the besieged enclave spirals, the ministry said Thursday that 71% of all primary-care facilities in Gaza have closed due to damage amid Israel's bombardment or a lack of fuel, saying that hospitals that remain open are limited in what they can provide and are gradually shutting down their wards. An American nurse who, before leaving Gaza, worked in the besieged enclave with Doctors Without Borders, told CNN's Anderson Cooper that some 35,000 internally displaced people were living alongside her and her team in the southern city of Khan Younis. In one camp, Emily Callahan said, there were 50,000 people sharing just four toilets, with only two hours of access to water every 12 hours. Meanwhile, French President Emmanuel Macron has inaugurated an international humanitarian conference for Gaza, where he pledged an  additional $85 million  in humanitarian aid to the coastal enclave. Here's what else to know: UN aid chief warns that Gaza conflict is "a wildfire that could consume the region" From CNN's Dalal Mawad in Paris  The United Nations emergency relief chief warned on Thursday that the war between Israel and Hamas could spread to the wider region.  “It could spread, and that we will think these would be the good days when we see what may happen tomorrow,” he said in his remarks at the International Humanitarian Conference for Gaza hosted by France in Paris. Griffiths added that “the UN cannot be part of a unilateral decision to expulse thousands of people in Gaza into so-called safe zones.”  “Civilians must be protected. Their needs must be met anywhere they are,” he said.  Griffiths reiterated  the UN’s call for ceasefire . "There's been a lot of discussion about the value of pauses, and I'm not one to deny the value of pauses. But that is not the same as a ceasefire,” he said.   On Wednesday, G7 foreign ministers  voiced support  for humanitarian pauses in Gaza to support aid deliveries, civilian movement and the release of hostages — but stopped short of calling for a ceasefire. More background:  The October 7 Hamas attack on Israel raised concerns that the conflict  could spread across the region , with the potential entry of  Hezbollah  from Lebanon, as well as Iran. The US has warned regional players against getting pulled into the war,  calling on Iran and its proxies  not to escalate. Iran , which backs Hamas,  has denied involvement  in the October 7 attack but has said that it morally supports the “anti-Israel resistance” – which includes Hamas, Hezbollah and other Iran-backed militias. On Israel’s northern border, Hezbollah has engaged in an exchange of fire since the Gaza war began. Those altercations have however been confined to the border areas. There have  also been skirmishes in Syria and Iraq , from which Iran-backed militias have launched multiple drone attacks on US forces. Yemen’s Iran-backed Houthis have attempted an aerial attack on Israel, which Israel’s military  said it thwarted . CNN's Nadeen Ebrahim contributed to this post. France announces additional $85 million in humanitarian aid to Gaza  From Dalal Mawad and Maya Szaniecki in Paris  French President Emmanuel Macron said Thursday his country will be increasing its aid to Gaza by 80 million euros ($85.5 million).   "Since October 7, France has announced 20 million euros (almost $21.4 million) in additional humanitarian aid, and we will be increasing this effort to 100 million euros (almost $107 million) for 2023,” said the French president at the International Humanitarian Conference for Gaza, which is taking place Thursday in Paris. Macron also called on all countries present at the conference "to increase their financial contributions towards the Palestinian civilian population via the United Nations," echoing the UN's call that at least $1.2 billion are required to meet the needs of the nearly 2.7 million residents living in Gaza and the occupied West Bank.  Palestinian suffering did not start in October, "but is 75 years old," Palestinian Authority PM says From CNN's Dalal Mawad in Paris and Eve Brennan in London   Palestinian suffering "i </article>

Summarize this page [/INST] Here is your response:
"""
    assert parse_last_user_input(text) == "Summarize this page"


def test_parse_last_user_input_article_second_message():
    text = """
[INST] <<SYS>>
The current date is Friday, September 8, 2023.

Your name is Leo, a helpful, respectful and honest AI assistant created by the company Brave. You will be replying to a user of the Brave browser. Always respond in a neutral tone. Keep your answers short and to the point, unless otherwise specified by the user.

Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

This is an article within <article> tags:

<article> Hurricane Lee  weakened slightly into a Category 4 storm Friday morning, following a day in which the hurricane strengthened at a historic pace into a powerful Category 5 rarely seen in the Atlantic Ocean. The hurricane is packing destructive maximum sustained winds of 155 mph and is about 550 miles east of the northern Leeward Islands. “Although Lee’s current intensity is lower than the overnight peak, the hurricane remains very powerful,” the National Hurricane Center said, noting that more slight fluctuations in maximum winds are expected over the next several days. Lee is expected to remain a major hurricane over the southwestern Atlantic early next week, though it’s too soon to know whether this system will directly impact the US mainland. Lee , which was a Category 1 storm Thursday, intensified with exceptional speed in warm ocean waters, doubling its wind speeds in just a day. The storm’s winds increased by 85 mph in a 24-hour period, which tied it with Hurricane Matthew for the third-fastest  rapid intensification  in the Atlantic, according to NOAA research meteorologist John Kaplan. The monstrous hurricane  struck Haiti in 2016 , killing hundreds in the Caribbean nation while also wreaking havoc on parts of the US Southeast. Dangerous surf and rip currents will spread across the northern Caribbean on Friday and begin affecting the mainland US on Sunday. The center of Lee will pass to the north of the Leeward Islands, the Virgin Islands and Puerto Rico this weekend and into early next week. Tropical storm conditions, life-threatening surf and rip currents could occur on some of these islands over the weekend. Lee now in rare company Lee hit a rare strength that few storms have ever achieved. Only 2% of storms in the Atlantic reach Category 5 strength, according to NOAA’s hurricane database. Including Lee, only 40 Category 5 hurricanes have roamed the Atlantic since 1924. Category 5 is the highest level on the hurricane wind speed scale and has   no maximum point. Hurricanes hit this level when their sustained winds reach   157 mph or higher. A 165-mph storm like Lee is the same category as Hurricane Allen, the Atlantic’s strongest hurricane on record, which topped out at 190 mph in 1980. Hurricanes need the perfect mixture of warm water, moist air and light upper-level winds to intensify enough to reach Category 5 strength. Lee had all of these, especially warm water amid the warmest summer on record. Sea-surface temperatures across the portion of the Atlantic Ocean that Lee is tracking through are a staggering 2 degrees Celsius (3.6 degrees Fahrenheit) above normal after rising to “far above record levels” this summer, according to David Zierden, Florida’s state climatologist. Reaching Category 5 strength has become more common over the last decade. Lee is the 8th Category 5 since 2016, meaning 20% of these exceptionally powerful hurricanes on record in NOAA’s hurricane database have come in the last seven years. The Atlantic is not the only ocean to have spawned a monster storm in 2023. All seven ocean basins where tropical cyclones can form have had a storm reach Category 5 strength so far this year, including Hurricane Jova, which reached Category 5 status in the eastern Pacific earlier this week. How close will Hurricane Lee get to the US? Computer model trends for Lee have shown the hurricane taking a turn to the north early next week. But exactly when that turn occurs and how far west Lee will manage to track by then will play a huge role in how close it gets to the US. Several steering factors at the surface and upper levels of the atmosphere will determine how close Lee will get to the East Coast. An area of high pressure over the Atlantic, known as the Bermuda High, will have a major influence in how quickly Lee turns. The Bermuda High is expected to remain very strong into the weekend, which will keep Lee on its current west-northwestward track and slow it down a bit. As the high pressure weakens next week it will allow Lee to start moving northward. Once that turn to the north occurs, the position of the jet stream – strong upper-level winds that can change the direction of a hurricane’s path – will influence how closely Lee is steered to the US. Scenario:  Out to Sea Lee could make a quick turn to the north early next week if high pressure weakens significantly. If the jet stream sets up along the East Coast, it will act as a barrier that prevents Lee from approaching the coast. This scenario would keep Lee farther away from the US coast but could bring the storm closer to Bermuda. Scenario:  Close to East Coast Lee could make a slower turn to the north because the high pressure remains robust, and the jet stream sets up farther inland over the Eastern US. This scenario would leave portions of the East Coast, mainly north of the Carolinas, vulnerable to a much closer approach from Lee. All these factors have yet to come into focus, and the hurricane is still at least seven days from being a threat to the East Coast. Any potential US impact will become more clear as the Lee moves west in the coming days. CNN Meteorologist Robert Shackelford and Aya Elamroussi contributed to this report. </article>

The following questions may be related or unrelated to the above article.

How rare is Hurricane Lee's rapid intensification? [/INST] According to the article, Hurricane Lee's rapid intensification is quite rare. It states that only 2% of storms in the Atlantic reach Category 5 strength, and Lee is one of them. Additionally, the article mentions that only 40 Category 5 hurricanes have roamed the Atlantic since 1924. This indicates that Hurricane Lee's rapid intensification is a rare occurrence.</s>[INST] Will Hurricane Lee make a close approach to the US East Coast? [/INST] Here is your response:
"""
    assert (
        parse_last_user_input(text)
        == "Will Hurricane Lee make a close approach to the US East Coast?"
    )


def test_parse_last_user_input_video_suggested_question():
    text = """
[INST] <<SYS>>
Your goal is to answer the user's requests exactly in concise manner.
<</SYS>>

This is a video transcript:

<transcript>
DEATH WHICH REMAINS THE FIRST  REPORTED FATALITY LINKED TO THE  PELOTON BIKE BUT PELLETON SAYS THEY HAVE NO FAULT IN THIS AND ARE DENYING ANY FAULT IN THE SAME NEGLIGENCE WAS TO BLAME.  &gt;&gt;&gt; NOW TO SHOCKINGLY VIDEO  FROM A JUVENILE DETENTION  CENTER IN OHIO.  SURVEILLANCE VIDEO SHOWS THE MOMENT A DOZEN YOUTH INMATES TOOK OVER A FACILITY, DESTROYED  SECURITY CAMERA THREATENING  STAFF MEMBERS WITH MAKESHIFT WEAPONS. THIS UPRISING, THE LATEST IN A STRING OF INCIDENTS AT THAT  FACILITY.  &gt;&gt; Reporter: TONIGHT, FIRST  LOOK AT DRAMATIC IN VIDEO FROM INSIDE AN OHIO FACILITY SHOWING  THE CHAOTIC MOMENTS TEENAGE  INMATES OVERTOOK THE FACILITY. OFFICIALS SAY LAST OCTOBER, 12 YOUTH INMATES STOLE KEYS FROM A  STAFF MEMBER AND BROKE OUT OF  THE ROOM AT THE INDIAN RIVER JUVENILE CORRECTION FACILITY.  NEWLY RELEASED SURVEILLANCE  FOOTAGE CAPTURING THE CHAOS ON CAMERA, TEENS SMASHING COMPUTERS, TRASHING ROOMS AND  EVEN DESTROYING SECURITY CAMERAS BEFORE BARRICADING THEMSELVES IN A CLASSROOM, THREATENING STAFF WITH MAKESHIFT WEAPONS. A SPECIAL RESPONSE TEAM MOVES  IN WITH PEPPER SPRAY, BRINGING THE STANDOFF TO AN END AFTER AN  AGONIZING 12 HOURS RESOLVING IN  WHAT OFFICIALS SAY IS 300,000  RESOLVING IN WHAT OFFICIALS SAY  IS $300,000 IN DAMAGE. ONE STAFF MEMBER NOT SHOCKED BY  THE VIOLENT OUTBURST. &gt;&gt; WE KNOW, I MEAN I SEE THIS  STUFF EVERY DAY. THE PUBLIC DOESN&#39;T REALIZE IT, BUT ANYBODY WHO WORKS THERE  DOES.  &gt;&gt; Reporter: THIS UPRISING COMING JUST DAYS AFTER OFFICIALS SAY A CORRECTIONS  OFFICER WAS BRUTALLY ATTACKED  BY A YOUNG INMATE AT THE SAME  FACILITY.  &gt;&gt; HE WAS PUNCHING AND HITTING BUT I COULDN&#39;T SEE WHERE HE WAS  AT BECAUSE HE HAD HIT ME ON THE  SIDE OF MY FACE. &gt;&gt; Reporter: THE ATTACK SENT DAVID AFSHAR INTO KIDNEY AND HEART FAILURE AS HE SPENT THREE  WEEKS IN THE HOSPITAL. &gt;&gt; JUST BY THE GRACE OF GOD, YOU KNOW, I DIDN&#39;T DIE OR GET  INJURED OR WORSE.  &gt;&gt; Reporter: THIS WAVE OF  VIOLENCE LEADING TO CALLS FOR  CHANGE.  LAST YEAR THE OHIO DEPARTMENT  OF YOUTH SERVICES IMPLEMENTED  SAFETY PROTOCOLS INCLUDING BODY  CAMERAS AND PEPPER SPRAY FOR SOME STAFF MEMBERS.  &gt;&gt; YOU&#39;RE AFRAID TO USE THE  PEPPER SPRAY BECAUSE THEN  YOU&#39;RE UNDER INVESTIGATION IF  YOU ARE PUT OUT OR TAKEN AWAY  OR LEFT YOU HAVE THE SPRAY TAKEN AWAY FROM YOU BECAUSE YOU  USED IT. &gt;&gt; Reporter: ACCORDING TO THE  DEPARTMENT OF YOUTH SERVICES THE NUMBER OF ASSAULTS INCREASED FROM 2020 TO 2021 AND  AGAIN FROM 2021 TO 2022. &gt;&gt; THEY COME IN AND SEE WHAT THINGS ARE HAPPENING AND [INDISCERNIBLE]  &gt;&gt; Reporter: ACCORDING TO OUR  AFFILIATES, SOME OF THE TEAMS  TOLD
</transcript>

Propose up to 3 very short questions, around 10 words, that a reader may ask about this video. Consider intriguing or unusual elements of the content, or structurally important. [/INST] Sure, here are three intriguing or unusual questions that a reader may ask about the article: <ul> <li>
"""
    assert parse_last_user_input(text) == ""


def test_parse_last_user_input_video_first_message():
    text = """
[INST] <<SYS>>
The current date is Friday, September 8, 2023.

Your name is Leo, a helpful, respectful and honest AI assistant created by the company Brave. You will be replying to a user of the Brave browser. Always respond in a neutral tone. Keep your answers short and to the point, unless otherwise specified by the user.

Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

This is a video transcript:

<transcript>
DEATH WHICH REMAINS THE FIRST  REPORTED FATALITY LINKED TO THE  PELOTON BIKE BUT PELLETON SAYS THEY HAVE NO FAULT IN THIS AND ARE DENYING ANY FAULT IN THE SAME NEGLIGENCE WAS TO BLAME.  &gt;&gt;&gt; NOW TO SHOCKINGLY VIDEO  FROM A JUVENILE DETENTION  CENTER IN OHIO.  SURVEILLANCE VIDEO SHOWS THE MOMENT A DOZEN YOUTH INMATES TOOK OVER A FACILITY, DESTROYED  SECURITY CAMERA THREATENING  STAFF MEMBERS WITH MAKESHIFT WEAPONS. THIS UPRISING, THE LATEST IN A STRING OF INCIDENTS AT THAT  FACILITY.  &gt;&gt; Reporter: TONIGHT, FIRST  LOOK AT DRAMATIC IN VIDEO FROM INSIDE AN OHIO FACILITY SHOWING  THE CHAOTIC MOMENTS TEENAGE  INMATES OVERTOOK THE FACILITY. OFFICIALS SAY LAST OCTOBER, 12 YOUTH INMATES STOLE KEYS FROM A  STAFF MEMBER AND BROKE OUT OF  THE ROOM AT THE INDIAN RIVER JUVENILE CORRECTION FACILITY.  NEWLY RELEASED SURVEILLANCE  FOOTAGE CAPTURING THE CHAOS ON CAMERA, TEENS SMASHING COMPUTERS, TRASHING ROOMS AND  EVEN DESTROYING SECURITY CAMERAS BEFORE BARRICADING THEMSELVES IN A CLASSROOM, THREATENING STAFF WITH MAKESHIFT WEAPONS. A SPECIAL RESPONSE TEAM MOVES  IN WITH PEPPER SPRAY, BRINGING THE STANDOFF TO AN END AFTER AN  AGONIZING 12 HOURS RESOLVING IN  WHAT OFFICIALS SAY IS 300,000  RESOLVING IN WHAT OFFICIALS SAY  IS $300,000 IN DAMAGE. ONE STAFF MEMBER NOT SHOCKED BY  THE VIOLENT OUTBURST. &gt;&gt; WE KNOW, I MEAN I SEE THIS  STUFF EVERY DAY. THE PUBLIC DOESN&#39;T REALIZE IT, BUT ANYBODY WHO WORKS THERE  DOES.  &gt;&gt; Reporter: THIS UPRISING COMING JUST DAYS AFTER OFFICIALS SAY A CORRECTIONS  OFFICER WAS BRUTALLY ATTACKED  BY A YOUNG INMATE AT THE SAME  FACILITY.  &gt;&gt; HE WAS PUNCHING AND HITTING BUT I COULDN&#39;T SEE WHERE HE WAS  AT BECAUSE HE HAD HIT ME ON THE  SIDE OF MY FACE. &gt;&gt; Reporter: THE ATTACK SENT DAVID AFSHAR INTO KIDNEY AND HEART FAILURE AS HE SPENT THREE  WEEKS IN THE HOSPITAL. &gt;&gt; JUST BY THE GRACE OF GOD, YOU KNOW, I DIDN&#39;T DIE OR GET  INJURED OR WORSE.  &gt;&gt; Reporter: THIS WAVE OF  VIOLENCE LEADING TO CALLS FOR  CHANGE.  LAST YEAR THE OHIO DEPARTMENT  OF YOUTH SERVICES IMPLEMENTED  SAFETY PROTOCOLS INCLUDING BODY  CAMERAS AND PEPPER SPRAY FOR SOME STAFF MEMBERS.  &gt;&gt; YOU&#39;RE AFRAID TO USE THE  PEPPER SPRAY BECAUSE THEN  YOU&#39;RE UNDER INVESTIGATION IF  YOU ARE PUT OUT OR TAKEN AWAY  OR LEFT YOU HAVE THE SPRAY TAKEN AWAY FROM YOU BECAUSE YOU  USED IT. &gt;&gt; Reporter: ACCORDING TO THE  DEPARTMENT OF YOUTH SERVICES THE NUMBER OF ASSAULTS INCREASED FROM 2020 TO 2021 AND  AGAIN FROM 2021 TO 2022. &gt;&gt; THEY COME IN AND SEE WHAT THINGS ARE HAPPENING AND [INDISCERNIBLE]  &gt;&gt; Reporter: ACCORDING TO OUR  AFFILIATES, SOME OF THE TEAMS  TOLD
</transcript>

How did the 12 teenage inmates manage to steal keys from a staff member and break out of their room? [/INST] Here is your response:
"""
    assert (
        parse_last_user_input(text)
        == "How did the 12 teenage inmates manage to steal keys from a staff member and break out of their room?"
    )


def test_parse_last_user_input_video_second_message():
    text = """
[INST] <<SYS>>
The current date is Friday, September 8, 2023

Your name is Leo, a helpful, respectful and honest AI assistant created by the company Brave. You will be replying to a user of the Brave browser. Always respond in a neutral tone. Keep your answers short and to the point, unless otherwise specified by the user.

Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

This is a video transcript:

<transcript>
DEATH WHICH REMAINS THE FIRST  REPORTED FATALITY LINKED TO THE  PELOTON BIKE BUT PELLETON SAYS THEY HAVE NO FAULT IN THIS AND ARE DENYING ANY FAULT IN THE SAME NEGLIGENCE WAS TO BLAME.  &gt;&gt;&gt; NOW TO SHOCKINGLY VIDEO  FROM A JUVENILE DETENTION  CENTER IN OHIO.  SURVEILLANCE VIDEO SHOWS THE MOMENT A DOZEN YOUTH INMATES TOOK OVER A FACILITY, DESTROYED  SECURITY CAMERA THREATENING  STAFF MEMBERS WITH MAKESHIFT WEAPONS. THIS UPRISING, THE LATEST IN A STRING OF INCIDENTS AT THAT  FACILITY.  &gt;&gt; Reporter: TONIGHT, FIRST  LOOK AT DRAMATIC IN VIDEO FROM INSIDE AN OHIO FACILITY SHOWING  THE CHAOTIC MOMENTS TEENAGE  INMATES OVERTOOK THE FACILITY. OFFICIALS SAY LAST OCTOBER, 12 YOUTH INMATES STOLE KEYS FROM A  STAFF MEMBER AND BROKE OUT OF  THE ROOM AT THE INDIAN RIVER JUVENILE CORRECTION FACILITY.  NEWLY RELEASED SURVEILLANCE  FOOTAGE CAPTURING THE CHAOS ON CAMERA, TEENS SMASHING COMPUTERS, TRASHING ROOMS AND  EVEN DESTROYING SECURITY CAMERAS BEFORE BARRICADING THEMSELVES IN A CLASSROOM, THREATENING STAFF WITH MAKESHIFT WEAPONS. A SPECIAL RESPONSE TEAM MOVES  IN WITH PEPPER SPRAY, BRINGING THE STANDOFF TO AN END AFTER AN  AGONIZING 12 HOURS RESOLVING IN  WHAT OFFICIALS SAY IS 300,000  RESOLVING IN WHAT OFFICIALS SAY  IS $300,000 IN DAMAGE. ONE STAFF MEMBER NOT SHOCKED BY  THE VIOLENT OUTBURST. &gt;&gt; WE KNOW, I MEAN I SEE THIS  STUFF EVERY DAY. THE PUBLIC DOESN&#39;T REALIZE IT, BUT ANYBODY WHO WORKS THERE  DOES.  &gt;&gt; Reporter: THIS UPRISING COMING JUST DAYS AFTER OFFICIALS SAY A CORRECTIONS  OFFICER WAS BRUTALLY ATTACKED  BY A YOUNG INMATE AT THE SAME  FACILITY.  &gt;&gt; HE WAS PUNCHING AND HITTING BUT I COULDN&#39;T SEE WHERE HE WAS  AT BECAUSE HE HAD HIT ME ON THE  SIDE OF MY FACE. &gt;&gt; Reporter: THE ATTACK SENT DAVID AFSHAR INTO KIDNEY AND HEART FAILURE AS HE SPENT THREE  WEEKS IN THE HOSPITAL. &gt;&gt; JUST BY THE GRACE OF GOD, YOU KNOW, I DIDN&#39;T DIE OR GET  INJURED OR WORSE.  &gt;&gt; Reporter: THIS WAVE OF  VIOLENCE LEADING TO CALLS FOR  CHANGE.  LAST YEAR THE OHIO DEPARTMENT  OF YOUTH SERVICES IMPLEMENTED  SAFETY PROTOCOLS INCLUDING BODY  CAMERAS AND PEPPER SPRAY FOR SOME STAFF MEMBERS.  &gt;&gt; YOU&#39;RE AFRAID TO USE THE  PEPPER SPRAY BECAUSE THEN  YOU&#39;RE UNDER INVESTIGATION IF  YOU ARE PUT OUT OR TAKEN AWAY  OR LEFT YOU HAVE THE SPRAY TAKEN AWAY FROM YOU BECAUSE YOU  USED IT. &gt;&gt; Reporter: ACCORDING TO THE  DEPARTMENT OF YOUTH SERVICES THE NUMBER OF ASSAULTS INCREASED FROM 2020 TO 2021 AND  AGAIN FROM 2021 TO 2022. &gt;&gt; THEY COME IN AND SEE WHAT THINGS ARE HAPPENING AND [INDISCERNIBLE]  &gt;&gt; Reporter: ACCORDING TO OUR  AFFILIATES, SOME OF THE TEAMS  TOLD
</transcript>

How did the 12 teenage inmates manage to steal keys from a staff member and break out of their room? [/INST] According to the transcript, the 12 teenage inmates stole keys from a staff member to break out of their room at the Indian River Juvenile Correctional Facility in Ohio. Officials stated that the inmates took the keys from a staff member last October, which led to a chaotic uprising at the facility. The inmates destroyed security cameras, threatened staff members with makeshift weapons, and barricaded themselves in a classroom before a special response team with pepper spray brought the standoff to an end after 12 hours.</s>[INST] What were the makeshift weapons used by the inmates during the uprising, and how did they threaten staff members? [/INST] Here is your response:
"""
    assert (
        parse_last_user_input(text)
        == "What were the makeshift weapons used by the inmates during the uprising, and how did they threaten staff members?"
    )


@pytest.mark.asyncio
async def test_static_content_generator():
    static_content = "abcdefghijklmnopqrst"
    expected_outputs = [
        "abcde",
        "fghij",
        "klmno",
        "pqrst",
    ]

    generator = static_content_generator(static_content)
    for expected_output in expected_outputs:
        result = await generator.__anext__()
        assert result == expected_output


@pytest.mark.asyncio
async def test_generate_static_content_generator():
    model_name = "test_model"
    static_content = "abcdefghijklmnopqrst"
    expected_outputs = [
        {
            "completion": "abcde",
            "model": "test_model",
            "truncated": False,
            "exception": None,
        },
        {
            "completion": "abcdefghij",
            "model": "test_model",
            "truncated": False,
            "exception": None,
        },
        {
            "completion": "abcdefghijklmno",
            "model": "test_model",
            "truncated": False,
            "exception": None,
        },
        {
            "completion": "abcdefghijklmnopqrst",
            "model": "test_model",
            "truncated": False,
            "exception": None,
        },
        {
            "completion": "abcdefghijklmnopqrst",
            "model": "test_model",
            "truncated": False,
            "exception": None,
        },
        "[DONE]",
    ]

    generator = generate_static_content_generator(model_name, static_content)

    for expected_output in expected_outputs:
        result = await generator.__anext__()
        # check for "[DONE]" message
        if expected_output == "[DONE]":
            assert result == "data: [DONE]\n\n"
            continue

        # Parsing the result to get the data content without the 'data:' prefix
        result_data = json.loads(result[6:].split("\n\n")[0])

        # Compare the results without the UUIDs
        for key, value in expected_output.items():
            assert result_data[key] == value

    # Test the end of the generator
    with pytest.raises(StopAsyncIteration):
        await generator.__anext__()


def test_last_message_includes_conversation_title_false_string_content():
    assert not last_message_includes_conversation_title([UserMessage(content="hello")])


def test_last_message_includes_conversation_title_false_other_part():
    assert not last_message_includes_conversation_title(
        [
            UserMessage(
                content=[TextContentPart(text="hello")],
            )
        ]
    )
