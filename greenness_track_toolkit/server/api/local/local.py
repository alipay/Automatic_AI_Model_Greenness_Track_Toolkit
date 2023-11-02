import os

from flask import Flask, render_template, request, send_from_directory
from greenness_track_toolkit.server.service.local_emission import LocalEmissionService

app = Flask(
    import_name=__name__,
    template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates"),
    static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
)

local_emission = LocalEmissionService()


@app.route('/')
def list_local_log_file():
    args = request.args
    pageNum = int(args.get("pageNum", 1))
    pageSize = int(args.get("pageSize", 10))
    rep = local_emission.select_all_db_file(pageNum=pageNum, pageSize=pageSize)

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
    req = {
        'pageNum': pageNum,
        'pageSize': pageSize,
    }
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
    return render_template("DashboardLocal/index.html", rep=rep)


@app.route('/<idx>')
def experiment_detail(idx):
    rep = local_emission.select_emission_by_filename(int(idx))
    return render_template("Detail/index.html", rep=rep)