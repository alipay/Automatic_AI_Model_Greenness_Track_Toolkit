import datetime
import os

from flask import Flask, render_template, request, send_from_directory
from greenness_track_toolkit.server.service.experiment import ExperimentService, ExperimentListRequest
from greenness_track_toolkit import utils

app = Flask(
    import_name=__name__,
    template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates"),
    static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
)

experiment_service = ExperimentService()




@app.route('/')
def experiment_list():
    # args format
    args = request.args
    start_time = args.get("startTime", None)
    if start_time is None or start_time == '':
        start_time = "19000101000000.000000"
    else:
        start_time = utils.to_str_time(utils.to_str_time_args(start_time))
    end_time = args.get("endTime", None)
    if end_time is None or end_time == '':
        end_time = utils.to_str_time(datetime.datetime.now())
    else:
        end_time = utils.to_str_time(utils.to_str_time_args(end_time))
    req = ExperimentListRequest(
        pageSize=int(args.get("pageSize", 10)),
        pageNum=int(args.get("pageNum", 1)),
        startTime=start_time,
        endTime=end_time,
        owner=args.get("owner", "")
    )
    rep = experiment_service.select_by_page(
        request=req
    )

    # computation page
    total_page = rep.total // rep.page_size
    if rep.total % rep.page_size != 0:
        total_page += 1

    default_show_page = 5
    start_page = rep.page_no - 2
    show_pages = default_show_page

    if total_page < show_pages:
        show_pages = total_page
    print_page = []
    if start_page <= 0:
        start_page = 1

    if start_page + show_pages > total_page:
        start_page = total_page - show_pages + 1
    for i in range(start_page, start_page + show_pages):
        print_page.append(i)

    # querying data format
    req.startTime = utils.to_str_time_args_to_index(utils.format_to_time(req.startTime))
    req.endTime = utils.to_str_time_args_to_index(utils.format_to_time(req.endTime))
    rep = {
        "hasNextPage": rep.page_no != total_page,
        "hasPrePage": rep.page_no != 1,
        "showFirstPage": rep.page_no >= default_show_page - 1 and total_page > default_show_page,
        "showLastPage": rep.page_no < (total_page - default_show_page // 2) and total_page > default_show_page,
        "totalPage": total_page,
        "printPage": print_page,
        "data": rep,
        "req": req
    }
    print(rep)
    return render_template("Dashboard/index.html", rep=rep)


@app.route('/<eid>')
def experiment_detail(eid):
    rep = experiment_service.select_experiment_by_eid(eid)
    return render_template("Detail/index.html", rep=rep)